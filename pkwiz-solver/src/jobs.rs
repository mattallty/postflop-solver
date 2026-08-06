//! The job queue: one worker, a FIFO of spots, and a cancel flag per job.
//!
//! # Why one worker
//!
//! The engine is already parallel — `postflop-solver` uses rayon across the whole machine — and a
//! tree is measured in gigabytes. Running two solves at once would halve the throughput of each
//! and double the peak memory for no gain, so the sidecar runs them one at a time and queues the
//! rest. Concurrency here is not a performance feature, it is only the property that the session
//! keeps answering `progress` and `cancel` while a solve is running.
//!
//! # Why the game moves at the end
//!
//! The worker owns the [`PostFlopGame`] for the duration of the solve and only publishes it into
//! the job when the loop is over. Sharing it behind a lock the solve holds for minutes would mean
//! every `node` query blocking on the solve it is trying to watch. The consequence is deliberate
//! and documented: `node` on a running job is an error, not a wait.
//!
//! # Why a finished job does not keep its tree
//!
//! A solved tree is the largest thing this process holds — gigabytes, against a default ceiling of
//! four per tree — and the job list only ever grows. Keeping every game a session has solved would
//! mean the third flop solve of an afternoon failing on memory because of the first two, which are
//! sitting on disk anyway.
//!
//! So a job whose solution has been written to a file is **recoverable**, and a recoverable job's
//! game can be dropped and reloaded on demand. Starting a solve *or opening a file* releases every
//! other recoverable game first, which bounds resident trees to the one being acquired plus those
//! that were never saved. `node` reloads transparently, so nothing above this module can tell the
//! difference — only [`JobStatus::resident`] reveals it, for a host that wants to show it. A tree
//! that was never saved can still be freed, by [`Jobs::forget`]: the host declares it is done with
//! the answer, and the row goes with the tree.
//!
//! Both moments matter, and only one of them is obvious. A solution browser walking a library calls
//! `open` and never `solve`, so sweeping on `solve` alone would leave the commonest path — and
//! opening the same file twice — accumulating a tree per call with nothing to give it back.
//!
//! The engine's own `free_memory()` is the wrong tool here: it releases the storage but keeps the
//! tree, and reading a released job needs a *solved* game either way. Dropping the whole game
//! frees the tree too, and the [`Spot`] we kept can rebuild it — `engine::estimate` exists
//! precisely because constructing a tree is cheap next to solving one.

use std::collections::{BTreeMap, VecDeque};
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use postflop_solver::{Action, PostFlopGame};
use serde::{Deserialize, Serialize};

use crate::engine::{self, EngineError, MemoryEstimate, Sample, Solved, Stopped};
use crate::spot::Spot;

/// Identifies a job for the life of the session. Never reused.
pub type JobId = u64;

/// Where a job is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Phase {
    /// Accepted, waiting for the worker.
    Queued,
    /// Constructing the tree and allocating memory. No iterations yet.
    Building,
    /// Iterating.
    Running,
    /// Finished, by convergence or by hitting the cap.
    Done,
    /// Stopped early. The strategy is finalized and readable, just further from equilibrium.
    Cancelled,
    /// Could not be built or blew up while solving. [`JobStatus::error`] says why.
    Failed,
}

impl Phase {
    /// Whether nothing further will happen to this job without a new command.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Done | Self::Cancelled | Self::Failed)
    }

    /// Whether the job holds a finalized game that can be inspected.
    #[must_use]
    pub const fn is_readable(self) -> bool {
        matches!(self, Self::Done | Self::Cancelled)
    }
}

/// Everything a host needs to draw a progress bar and then a result.
///
/// The same struct is the body of the `progress` response and of every pushed `job` event, so a
/// host that polls and a host that streams are reading identical data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobStatus {
    pub job_id: JobId,
    pub phase: Phase,
    pub iterations: u32,
    pub max_iterations: u32,
    /// `None` until the first measurement, which happens before iteration 1.
    pub exploitability: Option<f32>,
    pub target_exploitability: f32,
    /// Pot the target is relative to, so a host can render "0.4 / 1.0 (0.5% of 200)".
    pub starting_pot: i32,
    pub elapsed_ms: u64,
    /// Present from the end of the build onwards.
    pub memory: Option<MemoryEstimate>,
    /// Why the loop ended; present once terminal.
    pub stopped: Option<Stopped>,
    /// Root EV of each player, bias-subtracted. Present once the game is finalized.
    pub ev: Option<[f32; 2]>,
    pub saved_to: Option<String>,
    /// Whether the tree is in memory right now.
    ///
    /// `false` on a finished job with a [`Self::saved_to`] means it was released and will be
    /// reloaded from that file the next time it is read — informational, not something a host has
    /// to act on.
    pub resident: bool,
    pub error: Option<String>,
    /// The convergence curve, decimated (see `engine::push_sample`).
    pub history: Vec<Sample>,
}

impl JobStatus {
    fn new(job_id: JobId, spot: &Spot) -> Self {
        Self {
            job_id,
            phase: Phase::Queued,
            iterations: 0,
            max_iterations: spot.stop.max_iterations,
            exploitability: None,
            target_exploitability: spot.stop.target_for(spot.pot),
            starting_pot: spot.pot,
            elapsed_ms: 0,
            memory: None,
            stopped: None,
            ev: None,
            saved_to: None,
            resident: false,
            error: None,
            history: Vec::new(),
        }
    }
}

/// Where pushed frames go.
///
/// A trait rather than a channel so tests can collect events in a `Vec` and the binary can hand
/// them to the stdout writer, without either knowing about the other.
/// `Debug` is required so the types holding one can derive it — the crate warns on any public
/// type that cannot be printed, and a sink you cannot name is a bad thing to find in a log.
pub trait Emit: Send + Sync + std::fmt::Debug {
    /// Deliver one already-encoded JSON line. Must not block for long.
    fn emit(&self, line: String);
}

/// Discards everything. Useful when driving [`Jobs`] from a test that only polls.
#[derive(Debug, Clone, Copy, Default)]
pub struct Silent;

impl Emit for Silent {
    fn emit(&self, _line: String) {}
}

struct Job {
    spot: Spot,
    cancel: AtomicBool,
    status: Mutex<JobStatus>,
    /// Published by the worker once the solve is over, or by `open` for a loaded solution.
    game: Mutex<Option<PostFlopGame>>,
}

struct Shared {
    jobs: Mutex<BTreeMap<JobId, Arc<Job>>>,
    queue: Mutex<VecDeque<JobId>>,
    wake: Condvar,
    next_id: AtomicU64,
    stopping: AtomicBool,
    emit: Arc<dyn Emit>,
    /// Minimum gap between pushed progress frames.
    throttle: Duration,
}

/// The queue, its worker, and the jobs it has run.
///
/// Cloning is cheap and shares everything; the worker thread holds one clone of the inner state.
#[derive(Clone)]
pub struct Jobs {
    shared: Arc<Shared>,
}

impl std::fmt::Debug for Jobs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Jobs")
            .field("jobs", &self.shared.jobs.lock().map(|j| j.len()).ok())
            .finish_non_exhaustive()
    }
}

/// Why a command about a job could not be carried out.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JobError {
    #[error("no job {0}")]
    NoSuchJob(JobId),
    #[error("job {id} is {phase:?}; its strategy can only be read once it has finished or been cancelled")]
    NotReadable { id: JobId, phase: Phase },
    #[error("job {0} has no solution to save")]
    NothingToSave(JobId),
    #[error("job {0} has not been saved to a file, so releasing its tree would discard it; save it first, or use `forget` to discard it deliberately")]
    NotRecoverable(JobId),
    #[error("job {id} is {phase:?}; it can only be forgotten once it is finished, cancelled, or failed — cancel it first")]
    NotFinished { id: JobId, phase: Phase },
    #[error("{0}")]
    Engine(#[from] EngineError),
    #[error("action index {index} is out of range at that node, which offers {available}")]
    BadHistory { index: usize, available: usize },
    #[error("the node reached by that history is a {kind} node, which has no strategy")]
    NoStrategy { kind: &'static str },
}

impl JobError {
    /// Stable discriminant for the host to branch on.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::NoSuchJob(_) => "no_such_job",
            Self::NotReadable { .. } => "not_readable",
            Self::NothingToSave(_) => "nothing_to_save",
            Self::NotRecoverable(_) => "not_recoverable",
            Self::NotFinished { .. } => "not_finished",
            Self::Engine(_) => "engine",
            Self::BadHistory { .. } | Self::NoStrategy { .. } => "bad_node",
        }
    }
}

impl Jobs {
    /// Start the worker thread.
    ///
    /// The thread lives until [`Jobs::shutdown`] or the process exits.
    #[must_use]
    pub fn new(emit: Arc<dyn Emit>) -> Self {
        Self::with_throttle(emit, Duration::from_millis(200))
    }

    /// [`Jobs::new`], with the progress-frame rate limit set explicitly.
    ///
    /// Tests use a zero throttle so a two-second solve still produces frames to assert on.
    #[must_use]
    pub fn with_throttle(emit: Arc<dyn Emit>, throttle: Duration) -> Self {
        let shared = Arc::new(Shared {
            jobs: Mutex::new(BTreeMap::new()),
            queue: Mutex::new(VecDeque::new()),
            wake: Condvar::new(),
            next_id: AtomicU64::new(1),
            stopping: AtomicBool::new(false),
            emit,
            throttle,
        });

        let worker = Arc::clone(&shared);
        // Detached: the session ends by dropping the pipe, and a worker still finalizing a
        // cancelled tree must not be able to hold the process open.
        std::thread::Builder::new()
            .name("pkwiz-solver-worker".to_owned())
            .spawn(move || work(&worker))
            .expect("the OS must let us start one worker thread");

        Self { shared }
    }

    /// Accept a spot. Returns immediately with the queued job's status.
    ///
    /// # Errors
    ///
    /// If the spot does not validate. Everything that can be checked without building a tree is
    /// checked here, on the caller's thread, so a bad request fails the `solve` command rather
    /// than producing a job that fails a moment later.
    pub fn submit(&self, spot: Spot) -> Result<JobStatus, JobError> {
        spot.validate().map_err(EngineError::from)?;

        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let status = JobStatus::new(id, &spot);
        let job = Arc::new(Job {
            spot,
            cancel: AtomicBool::new(false),
            status: Mutex::new(status.clone()),
            game: Mutex::new(None),
        });

        self.shared.jobs.lock().unwrap().insert(id, job);
        // Emitted *before* the worker can see it. Otherwise a job that starts instantly races its
        // own arrival and the host receives `building` before `queued`, which reads as a bug in
        // whichever component is looked at second.
        self.shared.emit_status(&status);
        self.shared.queue.lock().unwrap().push_back(id);
        self.shared.wake.notify_all();

        Ok(status)
    }

    /// Current status of one job.
    ///
    /// # Errors
    ///
    /// If the id is unknown.
    pub fn status(&self, id: JobId) -> Result<JobStatus, JobError> {
        Ok(self.job(id)?.status.lock().unwrap().clone())
    }

    /// Every job this session has seen, oldest first.
    #[must_use]
    pub fn list(&self) -> Vec<JobStatus> {
        self.shared
            .jobs
            .lock()
            .unwrap()
            .values()
            .map(|j| j.status.lock().unwrap().clone())
            .collect()
    }

    /// Ask a job to stop.
    ///
    /// Returns as soon as the flag is set — it does *not* wait for the worker to notice. A job
    /// that has not started yet is cancelled outright; the worker skips it when it reaches the
    /// front of the queue. Cancelling a finished job is a no-op, not an error, because a UI
    /// racing the last progress frame should not have to care.
    ///
    /// # Errors
    ///
    /// If the id is unknown.
    pub fn cancel(&self, id: JobId) -> Result<JobStatus, JobError> {
        let job = self.job(id)?;
        job.cancel.store(true, Ordering::Relaxed);

        let mut status = job.status.lock().unwrap();
        let became_terminal = status.phase == Phase::Queued;
        if became_terminal {
            status.phase = Phase::Cancelled;
            status.stopped = Some(Stopped::Cancelled);
            // Emitted under the lock — see `Shared::emit_status`.
            self.shared.emit_status(&status);
        }
        Ok(status.clone())
    }

    /// Write a finished job's solution to disk.
    ///
    /// # Errors
    ///
    /// If the id is unknown, the job never produced a game, or the file cannot be written.
    pub fn save(&self, id: JobId, path: &str) -> Result<JobStatus, JobError> {
        let job = self.job(id)?;
        {
            let guard = job.game.lock().unwrap();
            let game = guard.as_ref().ok_or(JobError::NothingToSave(id))?;
            engine::save(
                game,
                path,
                job.spot.memo.as_deref().unwrap_or_default(),
                job.spot.compression_level,
            )?;
        }
        let mut status = job.status.lock().unwrap();
        status.saved_to = Some(path.to_owned());
        Ok(status.clone())
    }

    /// Drop a finished job's tree, keeping its row and its numbers.
    ///
    /// The job stays in the list, its status is untouched apart from
    /// [`JobStatus::resident`], and reading it again reloads the file. Releasing a job that is
    /// already released is a no-op rather than an error, so a host can call this without tracking
    /// what it has already done.
    ///
    /// # Errors
    ///
    /// If the id is unknown, the job has not finished, or it was never written to a file — in that
    /// last case the tree in memory is the only copy and dropping it would throw the solve away.
    pub fn release(&self, id: JobId) -> Result<JobStatus, JobError> {
        let job = self.job(id)?;
        let (phase, saved) = {
            let status = job.status.lock().unwrap();
            (status.phase, status.saved_to.is_some())
        };
        if !phase.is_readable() {
            return Err(JobError::NotReadable { id, phase });
        }
        if !saved {
            return Err(JobError::NotRecoverable(id));
        }

        release_recoverable(&self.shared, &job);
        let status = job.status.lock().unwrap().clone();
        Ok(status)
    }

    /// Remove a finished job entirely, dropping its tree and its row.
    ///
    /// This is the deliberate-discard counterpart to [`Jobs::release`]: `release` refuses a job
    /// that was never saved, because reading it again would have nothing to reload from — which
    /// left a session that solves without `savePath` pinning every tree it ever produced until
    /// the process exited. `forget` is the escape hatch: the host says it is done with the
    /// answer, saved or not, and both the tree and the row go. The id is never reused;
    /// afterwards the job answers `no_such_job`.
    ///
    /// No frame is pushed: the job's terminal frame was already emitted, and a row that no
    /// longer exists has no status to report. The removed job's final status is returned as the
    /// response instead.
    ///
    /// # Errors
    ///
    /// If the id is unknown, or the job is not yet terminal — cancel it first, so the worker is
    /// never left solving into a row that no longer exists.
    pub fn forget(&self, id: JobId) -> Result<JobStatus, JobError> {
        let mut jobs = self.shared.jobs.lock().unwrap();
        let job = jobs.get(&id).cloned().ok_or(JobError::NoSuchJob(id))?;
        // Only terminal jobs are removed, and a terminal phase never regresses, so the worker —
        // whose claim requires `Queued` — can never take a job that passed this check. A job
        // cancelled while queued may still have its id in the queue; the worker handles that by
        // looking the id up again and finding nothing.
        let status = job.status.lock().unwrap().clone();
        if !status.phase.is_terminal() {
            return Err(JobError::NotFinished {
                id,
                phase: status.phase,
            });
        }
        jobs.remove(&id);
        drop(jobs);
        // The `job` Arc dropped at the end of this scope is usually the last one, taking the
        // game — potentially gigabytes — with it, outside every lock.
        Ok(status)
    }

    /// Read a solution back from disk as a new, already-finished job.
    ///
    /// Reopening is the point of saving: a solution browser lists files and hands
    /// each one back to the same `node` command the live solve uses, so nothing downstream has to
    /// know whether a strategy was just computed or loaded.
    ///
    /// `max_memory_bytes` caps what the load may allocate; `None` means
    /// [`crate::DEFAULT_MEMORY_LIMIT`]. The solve path goes to great lengths to answer `tooBig`
    /// instead of being OOM-killed, and a browser opening an oversized (or corrupt-header) file
    /// deserves the same refusal. The cap is remembered on the job, so a later transparent
    /// reload obeys it too.
    ///
    /// # Errors
    ///
    /// If the file cannot be read, is not a solution, or needs more memory than the cap allows.
    pub fn open(&self, path: &str, max_memory_bytes: Option<u64>) -> Result<JobStatus, JobError> {
        // Opening is the other moment the process is about to hold a whole tree — a solution
        // browser walking a library calls this and never `solve`, so without this the sweep would
        // never run on the commonest path and each file opened, including the same file twice,
        // would add a tree that nothing gives back. Nothing is kept: the new job does not exist
        // yet. Done before the load rather than after, because the point is to hand memory back
        // before asking for more; a load that then fails has cost only some lazy reloading.
        release_others(&self.shared, None);

        let (game, memo) = engine::load(path, max_memory_bytes)?;

        let tree = game.tree_config();
        let spot = Spot {
            oop: crate::spot::RangeSpec::Notation(String::new()),
            ip: crate::spot::RangeSpec::Notation(String::new()),
            board: crate::spot::BoardSpec::Cards(
                game.current_board()
                    .iter()
                    .filter_map(|c| crate::convert::from_engine_card(*c).ok())
                    .map(|c| c.to_string())
                    .collect(),
            ),
            pot: tree.starting_pot,
            effective_stack: tree.effective_stack,
            sizing: crate::spot::Sizing::default(),
            rake: crate::spot::Rake {
                rate: tree.rake_rate,
                cap: tree.rake_cap,
            },
            stop: crate::spot::Stop::default(),
            compress: false,
            max_memory_bytes,
            save_path: None,
            compression_level: crate::spot::Spot::default_compression_level(),
            memo: Some(memo),
        };

        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let mut status = JobStatus::new(id, &spot);
        status.phase = Phase::Done;
        status.saved_to = Some(path.to_owned());
        status.resident = true;
        // A loaded solution carries no record of how it was reached: the engine serialises the
        // strategy, not the run. So every field that describes the *run* is left empty —
        // `stopped`, `exploitability`, `history`, and a `maxIterations` of zero rather than the
        // default a progress bar would divide by — and the fields that describe the *answer* are
        // recomputed from the tree we actually have.
        status.max_iterations = 0;
        status.target_exploitability = 0.0;
        status.ev = Some(postflop_solver::compute_current_ev(&game));

        let job = Arc::new(Job {
            spot,
            cancel: AtomicBool::new(false),
            status: Mutex::new(status.clone()),
            game: Mutex::new(Some(game)),
        });
        self.shared.jobs.lock().unwrap().insert(id, job);
        self.shared.emit_status(&status);

        Ok(status)
    }

    /// Inspect one node of a finished job's strategy.
    ///
    /// A job whose tree was released is reloaded from its file first, so a released job answers
    /// this exactly as it did before — more slowly, once.
    ///
    /// # Errors
    ///
    /// If the id is unknown, the job has not finished, the history does not describe a node, or the
    /// job was released and its file can no longer be read.
    pub fn node(&self, id: JobId, history: &[usize]) -> Result<NodeView, JobError> {
        let job = self.job(id)?;
        let (phase, saved_to) = {
            let status = job.status.lock().unwrap();
            (status.phase, status.saved_to.clone())
        };
        if !phase.is_readable() {
            return Err(JobError::NotReadable { id, phase });
        }

        let mut guard = job.game.lock().unwrap();
        if guard.is_none() {
            // Released, or never published. Only the first is recoverable. The reload honours
            // the same memory cap the job was allowed when it was built or opened.
            let path = saved_to.ok_or(JobError::NotReadable { id, phase })?;
            *guard = Some(engine::load(&path, job.spot.max_memory_bytes)?.0);
            // The status is updated while the game guard is still held: dropping it first
            // would let `release_others` (from a concurrent `open`, or a solve starting) take
            // the freshly loaded game in the gap, leaving `resident: true` with no game and
            // turning this read into a bogus error. No deadlock: this game → status nesting is
            // the one ordering the module uses, and no path acquires the game lock while
            // holding a status lock.
            job.status.lock().unwrap().resident = true;
        }
        let game = guard.as_mut().ok_or(JobError::NotReadable { id, phase })?;
        node_view(game, history)
    }

    /// Stop the worker after the job it is on. Idempotent.
    pub fn shutdown(&self) {
        self.shared.stopping.store(true, Ordering::Relaxed);
        self.shared.wake.notify_all();
    }

    fn job(&self, id: JobId) -> Result<Arc<Job>, JobError> {
        self.shared
            .jobs
            .lock()
            .unwrap()
            .get(&id)
            .cloned()
            .ok_or(JobError::NoSuchJob(id))
    }
}

impl Shared {
    /// Push one job frame.
    ///
    /// # The locking rule, which is load-bearing
    ///
    /// **A frame that makes a job terminal is emitted while its status lock is still held.**
    /// Everything else drops the lock first.
    ///
    /// Without that, a caller polling `progress` can see `done` *before* the terminal frame has
    /// been written, and a host that closes its stream on the polled answer loses the frame
    /// carrying the EVs and the convergence curve. Holding the lock across one pipe write costs
    /// nothing at the end of a job and makes "the poll says finished" imply "the last event has
    /// been sent". Progress frames deliberately do not pay that cost: there are many of them and
    /// a `progress` query must not queue behind one.
    fn emit_status(&self, status: &JobStatus) {
        let frame = serde_json::json!({ "event": "job", "job": status });
        // `to_string` on a value that came from `Serialize` cannot fail in practice; if it
        // somehow did, dropping one progress frame is better than killing the worker.
        if let Ok(line) = serde_json::to_string(&frame) {
            self.emit.emit(line);
        }
    }
}

/// Drop one job's game and say so. Assumes the caller has established that it is recoverable.
///
/// The game guard is released before the status lock is taken. Only `node`'s reload path nests
/// the two, in game → status order; nothing nests them the other way, which is what keeps the
/// module deadlock-free.
fn release_recoverable(shared: &Arc<Shared>, job: &Arc<Job>) {
    let dropped = job.game.lock().unwrap().take();
    if dropped.is_none() {
        // Already released. Nothing changed, so nothing is announced.
        return;
    }
    // Dropped outside both locks: freeing gigabytes is not instant and a `progress` query must not
    // queue behind it.
    drop(dropped);

    let mut status = job.status.lock().unwrap();
    status.resident = false;
    let snapshot = status.clone();
    drop(status);
    shared.emit_status(&snapshot);
}

/// Release every recoverable game, optionally sparing one.
///
/// Called as a solve starts and as a file is opened — the two moments the process is about to hold
/// a whole tree. `keep` is `None` when the job that is about to own one does not exist yet.
fn release_others(shared: &Arc<Shared>, keep: Option<JobId>) {
    let candidates: Vec<Arc<Job>> = shared
        .jobs
        .lock()
        .unwrap()
        .iter()
        .filter(|(&id, _)| Some(id) != keep)
        .map(|(_, job)| Arc::clone(job))
        .collect();

    for job in candidates {
        let recoverable = {
            let status = job.status.lock().unwrap();
            status.phase.is_readable() && status.saved_to.is_some() && status.resident
        };
        if recoverable {
            release_recoverable(shared, &job);
        }
    }
}

/// The worker loop: take the next job, run it, publish it, repeat.
fn work(shared: &Arc<Shared>) {
    loop {
        let next = {
            let mut queue = shared.queue.lock().unwrap();
            loop {
                if shared.stopping.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(id) = queue.pop_front() {
                    break id;
                }
                let (guard, _) = shared
                    .wake
                    .wait_timeout(queue, Duration::from_millis(100))
                    .unwrap();
                queue = guard;
            }
        };

        let Some(job) = shared.jobs.lock().unwrap().get(&next).cloned() else {
            continue;
        };

        run_job(shared, &job);
    }
}

fn run_job(shared: &Arc<Shared>, job: &Arc<Job>) {
    let started = Instant::now();

    // Claim the job: the cancel check and the Queued → Building transition happen inside one
    // critical section on the status lock. `cancel` marks a job terminal only while it is still
    // `Queued`, under the same lock — so either it wins and the worker sees a non-Queued phase
    // here and walks away, or the claim wins and `cancel` degrades to the set-the-flag path. A
    // bare flag check before the transition had a window in which a job could be marked
    // `Cancelled` (a terminal frame, emitted) and then hauled back to `Building` by the worker,
    // which both violates "terminal frames are final" and burns a full tree build.
    {
        let mut status = job.status.lock().unwrap();
        if job.cancel.load(Ordering::Relaxed) || status.phase != Phase::Queued {
            // Cancelled before it ever started; `cancel` already set the phase and emitted.
            return;
        }
        status.phase = Phase::Building;
        status.elapsed_ms = 0;
        let snapshot = status.clone();
        // Dropped before emitting: this frame is not terminal, and a `progress` query must not
        // have to wait behind a pipe write to be answered.
        drop(status);
        shared.emit_status(&snapshot);
    }

    // Before allocating: hand back every tree that is already on disk. This is the difference
    // between an afternoon of solves costing one tree of memory and costing all of them.
    let id = job.status.lock().unwrap().job_id;
    release_others(shared, Some(id));

    // The engine panics rather than returning an error for several kinds of misuse, and it is
    // full of `unsafe` in its hot loops. A panic here must fail one job, not wedge the queue and
    // silence every future solve.
    let built = std::panic::catch_unwind(AssertUnwindSafe(|| engine::build(&job.spot)));

    let (mut game, memory) = match built {
        Ok(Ok(pair)) => pair,
        Ok(Err(e)) => return fail(shared, job, started, e.to_string()),
        Err(payload) => return fail(shared, job, started, crate::describe_panic(&*payload)),
    };

    {
        let mut status = job.status.lock().unwrap();
        status.memory = Some(memory);
        status.phase = Phase::Running;
        status.elapsed_ms = started.elapsed().as_millis() as u64;
        let snapshot = status.clone();
        drop(status);
        shared.emit_status(&snapshot);
    }

    let mut last_emit = Instant::now();
    let solved = std::panic::catch_unwind(AssertUnwindSafe(|| {
        engine::run(
            &mut game,
            &job.spot.stop,
            job.spot.pot,
            &job.cancel,
            |iterations, exploitability| {
                let mut status = job.status.lock().unwrap();
                status.iterations = iterations;
                status.exploitability = Some(exploitability);
                status.elapsed_ms = started.elapsed().as_millis() as u64;
                // Throttled: a river tree measures exploitability many times a second and a
                // frame per measurement would drown the pipe in noise nobody can read.
                let due = last_emit.elapsed() >= shared.throttle;
                let snapshot = due.then(|| status.clone());
                drop(status);
                if let Some(snapshot) = snapshot {
                    last_emit = Instant::now();
                    shared.emit_status(&snapshot);
                }
            },
        )
    }));

    let solved: Solved = match solved {
        Ok(s) => s,
        Err(payload) => return fail(shared, job, started, crate::describe_panic(&*payload)),
    };

    let saved_to = job.spot.save_path.as_deref().and_then(|path| {
        // A failed save must not turn a finished solve into a failure — the answer is in memory
        // and still worth having — so it is reported in `error` while the phase stays `Done`.
        match engine::save(
            &game,
            path,
            job.spot.memo.as_deref().unwrap_or_default(),
            job.spot.compression_level,
        ) {
            Ok(()) => Some(path.to_owned()),
            Err(e) => {
                job.status.lock().unwrap().error = Some(e.to_string());
                None
            }
        }
    });

    *job.game.lock().unwrap() = Some(game);

    let mut status = job.status.lock().unwrap();
    status.phase = if solved.stopped == Stopped::Cancelled {
        Phase::Cancelled
    } else {
        Phase::Done
    };
    status.iterations = solved.iterations;
    status.exploitability = Some(solved.exploitability);
    status.stopped = Some(solved.stopped);
    status.ev = Some(solved.ev);
    status.history = solved.history;
    status.saved_to = saved_to;
    status.resident = true;
    status.elapsed_ms = started.elapsed().as_millis() as u64;
    // Emitted under the lock — see `Shared::emit_status`.
    shared.emit_status(&status);
}

fn fail(shared: &Arc<Shared>, job: &Arc<Job>, started: Instant, message: String) {
    let mut status = job.status.lock().unwrap();
    status.phase = Phase::Failed;
    status.error = Some(message);
    status.elapsed_ms = started.elapsed().as_millis() as u64;
    // Emitted under the lock — see `Shared::emit_status`.
    shared.emit_status(&status);
}

/// One node of a solved tree, in a form a range grid can render directly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeView {
    /// The action indices that were replayed to get here.
    pub history: Vec<usize>,
    /// The board at this node, which grows as chance nodes are played through.
    pub board: Vec<String>,
    pub is_terminal: bool,
    pub is_chance: bool,
    /// `0` for OOP, `1` for IP. Absent at terminal and chance nodes.
    pub player: Option<usize>,
    /// The actions available here, rendered (`"Check"`, `"Bet(120)"`, `"Chance(Qc)"`).
    pub actions: Vec<String>,
    /// The acting player's hands, in the order every other array here uses.
    pub hands: Vec<String>,
    /// Action-major: `strategy[a][h]` is how often hand `h` takes action `a`. Empty at a terminal
    /// or chance node.
    pub strategy: Vec<Vec<f32>>,
    /// Normalized weight of each hand, i.e. how much of the acting player's range it is here.
    pub weights: Vec<f32>,
    /// Equity of each hand at this node.
    pub equity: Vec<f32>,
    /// Expected value of each hand at this node.
    pub ev: Vec<f32>,
    /// Range-weighted averages of the two arrays above — the numbers a header line shows.
    pub average_equity: f32,
    pub average_ev: f32,
}

fn node_view(game: &mut PostFlopGame, history: &[usize]) -> Result<NodeView, JobError> {
    game.back_to_root();
    for (depth, index) in history.iter().enumerate() {
        if game.is_terminal_node() {
            return Err(JobError::BadHistory {
                index: *index,
                available: 0,
            });
        }
        if game.is_chance_node() {
            // At a chance node the "action" is the dealt card id, not an offset into a list.
            let mask = game.possible_cards();
            if *index >= 52 || mask & (1u64 << *index) == 0 {
                return Err(JobError::BadHistory {
                    index: *index,
                    available: mask.count_ones() as usize,
                });
            }
        } else {
            let available = game.available_actions().len();
            if *index >= available {
                return Err(JobError::BadHistory {
                    index: *index,
                    available,
                });
            }
        }
        let _ = depth;
        game.play(*index);
    }

    let is_terminal = game.is_terminal_node();
    let is_chance = game.is_chance_node();
    let board = game
        .current_board()
        .iter()
        .filter_map(|c| crate::convert::from_engine_card(*c).ok())
        .map(|c| c.to_string())
        .collect();

    let actions: Vec<String> = if is_chance {
        // The engine's chance "actions" are the 52 card ids, filtered by what is still live.
        let mask = game.possible_cards();
        (0..52u8)
            .filter(|c| mask & (1u64 << c) != 0)
            .filter_map(|c| crate::convert::from_engine_card(c).ok())
            .map(|c| format!("Chance({c})"))
            .collect()
    } else {
        game.available_actions().iter().map(render_action).collect()
    };

    if is_terminal || is_chance {
        return Ok(NodeView {
            history: history.to_vec(),
            board,
            is_terminal,
            is_chance,
            player: None,
            actions,
            hands: Vec::new(),
            strategy: Vec::new(),
            weights: Vec::new(),
            equity: Vec::new(),
            ev: Vec::new(),
            average_equity: f32::NAN,
            average_ev: f32::NAN,
        });
    }

    let player = game.current_player();
    game.cache_normalized_weights();

    let hands = game
        .private_cards(player)
        .iter()
        .map(|h| crate::convert::hole_to_string(*h).unwrap_or_else(|_| "??".to_owned()))
        .collect::<Vec<_>>();
    let num_hands = hands.len();

    let flat = game.strategy();
    let strategy: Vec<Vec<f32>> = flat.chunks(num_hands.max(1)).map(<[f32]>::to_vec).collect();

    let weights = game.normalized_weights(player).to_vec();
    let equity = game.equity(player);
    let ev = game.expected_values(player);
    let average_equity = postflop_solver::compute_average(&equity, &weights);
    let average_ev = postflop_solver::compute_average(&ev, &weights);

    Ok(NodeView {
        history: history.to_vec(),
        board,
        is_terminal,
        is_chance,
        player: Some(player),
        actions,
        hands,
        strategy,
        weights,
        equity,
        ev,
        average_equity,
        average_ev,
    })
}

fn render_action(action: &Action) -> String {
    match action {
        Action::None => "None".to_owned(),
        Action::Fold => "Fold".to_owned(),
        Action::Check => "Check".to_owned(),
        Action::Call => "Call".to_owned(),
        Action::Bet(n) => format!("Bet({n})"),
        Action::Raise(n) => format!("Raise({n})"),
        Action::AllIn(n) => format!("AllIn({n})"),
        Action::Chance(c) => crate::convert::from_engine_card(*c)
            .map_or_else(|_| "Chance(?)".to_owned(), |c| format!("Chance({c})")),
    }
}
