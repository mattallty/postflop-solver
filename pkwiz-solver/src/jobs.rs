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
//!
//! # Bunching jobs, and why the sweep skips them
//!
//! A `prepareBunching` job is the same lifecycle around a different computation: minutes of
//! table-building instead of minutes of CFR, 300 discrete progress steps instead of iterations,
//! and a ~62 MB result instead of a tree. The sweep above deliberately leaves that result alone:
//! it is two orders of magnitude smaller than a tree, and a queued solve is often about to need
//! exactly it — releasing it on every solve would mean reloading 62 MB per solve of a session
//! that prepared it precisely to reuse it.
//!
//! A solve that used bunching data keeps its `Arc` to it for the job's life, even after
//! `release` drops the tree: the engine's file format does not record the bunching effect, so a
//! reloaded game must have `set_bunching_effect` re-applied or it silently answers different
//! numbers. That retained `Arc` pins ~62 MB per *distinct* preparation (shared across every
//! solve that used it) until the solve jobs themselves are forgotten.

use std::collections::{BTreeMap, VecDeque};
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use postflop_solver::{Action, BunchingData, PostFlopGame};
use serde::{Deserialize, Serialize};

use crate::engine::{self, EngineError, MemoryEstimate, Sample, Solved, Stopped};
use crate::spot::{BunchingRef, BunchingSpec, Spot};

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

/// What kind of computation a job is running.
///
/// Always serialized, so a host can branch on it before touching kind-specific fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum JobKind {
    /// A CFR solve of one spot.
    Solve,
    /// A bunching-effect preparation.
    Bunching,
}

/// Progress and result of a bunching preparation — [`JobStatus::bunching`], on bunching jobs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BunchingStatus {
    /// 1–3. Named `stage` on the wire so it never collides with [`JobStatus::phase`].
    pub stage: u8,
    /// 0–100 within the current stage.
    pub stage_percent: u8,
    /// `((stage − 1)·100 + stagePercent) / 3` — coarse by declaration: stage 3's steps are
    /// lopsided by design, so this is a progress bar, not a clock.
    pub overall_percent: u8,
    /// How many fold players the preparation covers (1–4).
    pub fold_players: usize,
    /// The preparation's flop, sorted — what a solve's first three board cards must match.
    pub flop: Vec<String>,
    /// Bytes the finished data holds resident (~62 MB). Present once `done`.
    pub memory_bytes: Option<u64>,
}

/// Everything a host needs to draw a progress bar and then a result.
///
/// The same struct is the body of the `progress` response and of every pushed `job` event, so a
/// host that polls and a host that streams are reading identical data.
///
/// On a bunching job the solve-only fields hold inert values — `iterations` and
/// `maxIterations` 0, `targetExploitability` 0.0, `startingPot` 0, empty `history` — and
/// [`Self::bunching`] carries the progress; `stopped` stays `null` on `done` (why a *loop*
/// ended is a solve concept) and is `cancelled` on a cancelled preparation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobStatus {
    pub job_id: JobId,
    pub kind: JobKind,
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
    ///
    /// Populated live: every progress frame carries the curve so far, so a host can draw
    /// convergence while the solve runs rather than only after it. The terminal frame carries
    /// the authoritative final curve.
    pub history: Vec<Sample>,
    /// Preparation progress and result. `null` on solve jobs.
    pub bunching: Option<BunchingStatus>,
}

impl JobStatus {
    fn new(job_id: JobId, spot: &Spot) -> Self {
        Self {
            job_id,
            kind: JobKind::Solve,
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
            bunching: None,
        }
    }

    fn new_bunching(job_id: JobId, fold_players: usize, flop: Vec<String>) -> Self {
        Self {
            job_id,
            kind: JobKind::Bunching,
            phase: Phase::Queued,
            iterations: 0,
            max_iterations: 0,
            exploitability: None,
            target_exploitability: 0.0,
            starting_pot: 0,
            elapsed_ms: 0,
            memory: None,
            stopped: None,
            ev: None,
            saved_to: None,
            resident: false,
            error: None,
            history: Vec::new(),
            bunching: Some(BunchingStatus {
                stage: 1,
                stage_percent: 0,
                overall_percent: 0,
                fold_players,
                flop,
                memory_bytes: None,
            }),
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

/// What a job computes: a solve, or a bunching preparation. Internal — [`JobStatus`] stays the
/// one wire shape either way. The spot is boxed because it dwarfs the other variant.
enum Task {
    Solve(Box<Spot>),
    Bunching(BunchingSpec),
}

impl Task {
    /// Fields both kinds share, so `save`, `node` and the reload path need not match twice.
    fn memo(&self) -> &str {
        match self {
            Self::Solve(spot) => spot.memo.as_deref(),
            Self::Bunching(spec) => spec.memo.as_deref(),
        }
        .unwrap_or_default()
    }

    fn compression_level(&self) -> Option<i32> {
        match self {
            Self::Solve(spot) => spot.compression_level,
            Self::Bunching(spec) => spec.compression_level,
        }
    }

    fn max_memory_bytes(&self) -> Option<u64> {
        match self {
            Self::Solve(spot) => spot.max_memory_bytes,
            Self::Bunching(spec) => spec.max_memory_bytes,
        }
    }
}

struct Job {
    task: Task,
    cancel: AtomicBool,
    status: Mutex<JobStatus>,
    /// Published by the worker once the solve is over, or by `open` for a loaded solution.
    /// Always `None` on a bunching job.
    game: Mutex<Option<PostFlopGame>>,
    /// On a bunching job: the result, published at `done` (or by `open_bunching` at creation).
    /// On a solve job that used bunching: the `Arc` captured when the run started, retained for
    /// the job's life so the reload after `release` can re-apply the effect. Locked in the same
    /// position as `game` — before `status`, never after — so the module's deadlock-freedom
    /// argument covers it unchanged.
    bunching_data: Mutex<Option<Arc<BunchingData>>>,
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
    #[error("job {0} was cancelled before it started; it never produced a strategy")]
    NeverRan(JobId),
    #[error("{0}")]
    Engine(#[from] EngineError),
    #[error("action index {index} is out of range at that node, which offers {available}")]
    BadHistory { index: usize, available: usize },
    #[error("the node reached by that history is a {kind} node, which has no strategy")]
    NoStrategy { kind: &'static str },
    #[error("job {0} is a bunching preparation; it has no strategy to read")]
    NotBunchingReadable(JobId),
    #[error("job {id} is a solve, not a bunching preparation")]
    NotBunching { id: JobId },
    #[error("bunching preparation {id} is {phase:?} and has no data; prepare it again")]
    BunchingNotReady { id: JobId, phase: Phase },
    #[error("job {id} was solved with bunching data that could not be re-established: {reason}")]
    BunchingUnavailable { id: JobId, reason: String },
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
            Self::NeverRan(_) => "never_ran",
            Self::Engine(_) => "engine",
            Self::BadHistory { .. } | Self::NoStrategy { .. } => "bad_node",
            // Deliberately shares `not_readable`'s code: to a host, both mean "this job has no
            // strategy to show" — only the prose differs, and hosts branch on the code.
            Self::NotBunchingReadable(_) => "not_readable",
            Self::NotBunching { .. } => "not_bunching",
            Self::BunchingNotReady { .. } => "bunching_not_ready",
            Self::BunchingUnavailable { .. } => "bunching_unavailable",
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
    /// than producing a job that fails a moment later. For a spot referencing a bunching job by
    /// id, that includes the reference itself: the job must exist, be a preparation, not be
    /// terminal without data, and its flop must match the spot's first three board cards —
    /// a `path` reference defers all of that to the worker, which is where the file is read.
    pub fn submit(&self, spot: Spot) -> Result<JobStatus, JobError> {
        let validated = spot.validate().map_err(EngineError::from)?;

        if let Some(BunchingRef::Job { job_id }) = &spot.bunching {
            let prep = self.job(*job_id)?;
            let Task::Bunching(spec) = &prep.task else {
                return Err(JobError::NotBunching { id: *job_id });
            };
            let has_data = prep.bunching_data.lock().unwrap().is_some();
            let (phase, saved) = {
                let status = prep.status.lock().unwrap();
                (status.phase, status.saved_to.is_some())
            };
            // A running or queued preparation is fine — the FIFO guarantees it finishes before
            // this solve starts. Terminal with neither data nor a file can never serve one.
            if phase.is_terminal() && !has_data && !saved {
                return Err(JobError::BunchingNotReady { id: *job_id, phase });
            }
            // Only the first three board cards have to match, sorted — the engine's own rule,
            // checked here so the mismatch names the flops instead of failing the job later.
            let ours = sorted_flop(&validated.board);
            let theirs = spec
                .flop
                .cards()
                .map(|cards| sorted_flop(&cards))
                .unwrap_or_default();
            if ours != theirs {
                return Err(EngineError::Spot(crate::spot::SpotError::Bunching {
                    reason: format!(
                        "job {job_id} prepared flop {} but the spot's flop is {}",
                        theirs.join(""),
                        ours.join(""),
                    ),
                })
                .into());
            }
        }

        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let status = JobStatus::new(id, &spot);
        let job = Arc::new(Job {
            task: Task::Solve(Box::new(spot)),
            cancel: AtomicBool::new(false),
            status: Mutex::new(status.clone()),
            game: Mutex::new(None),
            bunching_data: Mutex::new(None),
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

    /// Accept a bunching preparation. Returns immediately with the queued job's status.
    ///
    /// The validated data is deliberately *not* kept for the worker: it is cheap to rebuild,
    /// and keeping it would strand state behind a job cancelled while still queued.
    ///
    /// # Errors
    ///
    /// If the spec does not validate — a bad flop or range, the engine's own refusals
    /// (suit-asymmetric, empty, more than four fold players), or a preparation whose peak
    /// memory exceeds the spec's cap. All synchronous, before anything is allocated.
    pub fn submit_bunching(&self, spec: BunchingSpec) -> Result<JobStatus, JobError> {
        let data = engine::validate_bunching(&spec)?;
        let fold_players = data.fold_ranges().len();
        let flop = render_flop(data.flop());
        drop(data);

        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let status = JobStatus::new_bunching(id, fold_players, flop);
        let job = Arc::new(Job {
            task: Task::Bunching(spec),
            cancel: AtomicBool::new(false),
            status: Mutex::new(status.clone()),
            game: Mutex::new(None),
            bunching_data: Mutex::new(None),
        });

        self.shared.jobs.lock().unwrap().insert(id, job);
        // Emitted before the worker can see it, for the same frame-ordering reason as `submit`.
        self.shared.emit_status(&status);
        self.shared.queue.lock().unwrap().push_back(id);
        self.shared.wake.notify_all();

        Ok(status)
    }

    /// Read prepared bunching data back from disk as a new, already-finished job.
    ///
    /// The mirror of [`Jobs::open`] for `.bunching` files: the new job is `done`, resident, and
    /// referencable by `{"jobId":…}` from any solve on its flop. No sweep runs — the sweep
    /// exists for gigabyte trees and skips bunching data anyway.
    ///
    /// # Errors
    ///
    /// If the file cannot be read, is not bunching data (a game file answers with the engine's
    /// "Data type is invalid"), or claims more memory than the cap allows.
    pub fn open_bunching(
        &self,
        path: &str,
        max_memory_bytes: Option<u64>,
    ) -> Result<JobStatus, JobError> {
        let (data, memo) = engine::load_bunching(path, max_memory_bytes)?;
        let fold_players = data.fold_ranges().len();
        let flop = render_flop(data.flop());

        // The fold ranges cannot be reconstructed from the data, so the spec is a placeholder;
        // it is never re-validated — the data itself is what this job holds and serves.
        let spec = BunchingSpec {
            fold_ranges: Vec::new(),
            flop: crate::spot::BoardSpec::Cards(flop.clone()),
            max_memory_bytes,
            save_path: None,
            compression_level: Spot::default_compression_level(),
            memo: Some(memo),
        };

        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let mut status = JobStatus::new_bunching(id, fold_players, flop);
        status.phase = Phase::Done;
        status.saved_to = Some(path.to_owned());
        status.resident = true;
        if let Some(bunching) = status.bunching.as_mut() {
            bunching.stage = 3;
            bunching.stage_percent = 100;
            bunching.overall_percent = 100;
            bunching.memory_bytes = Some(data.memory_usage());
        }

        let job = Arc::new(Job {
            task: Task::Bunching(spec),
            cancel: AtomicBool::new(false),
            status: Mutex::new(status.clone()),
            game: Mutex::new(None),
            bunching_data: Mutex::new(Some(Arc::new(data))),
        });
        self.shared.jobs.lock().unwrap().insert(id, job);
        self.shared.emit_status(&status);

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

    /// Write a finished job's product — a solution, or prepared bunching data — to disk.
    ///
    /// # Errors
    ///
    /// If the id is unknown, the job never produced anything (a cancelled preparation keeps
    /// nothing, unlike a cancelled solve), or the file cannot be written.
    pub fn save(&self, id: JobId, path: &str) -> Result<JobStatus, JobError> {
        let job = self.job(id)?;
        match &job.task {
            Task::Solve(_) => {
                let guard = job.game.lock().unwrap();
                let game = guard.as_ref().ok_or(JobError::NothingToSave(id))?;
                engine::save(game, path, job.task.memo(), job.task.compression_level())?;
            }
            Task::Bunching(_) => {
                let guard = job.bunching_data.lock().unwrap();
                let data = guard.as_ref().ok_or(JobError::NothingToSave(id))?;
                engine::save_bunching(data, path, job.task.memo(), job.task.compression_level())?;
            }
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
    /// `bunching` re-applies a bunching effect to the loaded game: the file format records
    /// nothing about bunching, so a solution solved with it reads back *without* it unless the
    /// caller says which data to apply. Legal to omit — the numbers are then non-bunching, and
    /// the file cannot tell anyone otherwise — which is why hosts are advised to record the
    /// preparation in the memo they save with. The ref is remembered on the job so transparent
    /// reloads re-apply it too.
    ///
    /// # Errors
    ///
    /// If the file cannot be read, is not a solution, needs more memory than the cap allows, or
    /// the bunching ref cannot be resolved or applied (wrong flop, not-ready data).
    pub fn open(
        &self,
        path: &str,
        max_memory_bytes: Option<u64>,
        bunching: Option<BunchingRef>,
    ) -> Result<JobStatus, JobError> {
        // Opening is the other moment the process is about to hold a whole tree — a solution
        // browser walking a library calls this and never `solve`, so without this the sweep would
        // never run on the commonest path and each file opened, including the same file twice,
        // would add a tree that nothing gives back. Nothing is kept: the new job does not exist
        // yet. Done before the load rather than after, because the point is to hand memory back
        // before asking for more; a load that then fails has cost only some lazy reloading.
        release_others(&self.shared, None);

        let (mut game, memo) = engine::load(path, max_memory_bytes)?;

        let data = match &bunching {
            None => None,
            Some(bref) => {
                let data = resolve_bunching(&self.shared, bref, max_memory_bytes)?;
                // The full re-apply, not bare `set_bunching_effect`: a loaded game's stored
                // EVs were restored by the decoder without bunching (see
                // `engine::reapply_bunching`) and must be recomputed under it.
                engine::reapply_bunching(&mut game, &data).map_err(EngineError::Build)?;
                Some(data)
            }
        };

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
            // A reopened file's locks and tree edits live in the game itself (the engine
            // serializes them); this placeholder spot is never used to rebuild.
            added_lines: Vec::new(),
            removed_lines: Vec::new(),
            locks: Vec::new(),
            // The ref, not the data: it is what the transparent reload path consults.
            bunching,
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
        // Computed after any bunching effect was applied, so the EVs match what `node` shows.
        status.ev = Some(postflop_solver::compute_current_ev(&game));

        let job = Arc::new(Job {
            task: Task::Solve(Box::new(spot)),
            cancel: AtomicBool::new(false),
            status: Mutex::new(status.clone()),
            game: Mutex::new(Some(game)),
            bunching_data: Mutex::new(data),
        });
        self.shared.jobs.lock().unwrap().insert(id, job);
        self.shared.emit_status(&status);

        Ok(status)
    }

    /// Inspect one node of a finished job's strategy.
    ///
    /// A job whose tree was released is reloaded from its file first, so a released job answers
    /// this exactly as it did before — more slowly, once. A released *bunching* solve also has
    /// its bunching effect re-applied on that reload, because the file carries no trace of it;
    /// the retained data (or, failing that, the preparation job or its file) is what makes the
    /// reloaded answers bit-identical instead of silently non-bunching.
    ///
    /// # Errors
    ///
    /// If the id is unknown, the job is a bunching preparation (no strategy to read), it has
    /// not finished, the history does not describe a node, the job was released and its file
    /// can no longer be read, or its bunching effect can no longer be re-established.
    pub fn node(&self, id: JobId, history: &[usize]) -> Result<NodeView, JobError> {
        let job = self.job(id)?;
        if matches!(job.task, Task::Bunching(_)) {
            return Err(JobError::NotBunchingReadable(id));
        }
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
            // No game and no file: a job cancelled while still queued. (`Done` always has one
            // or the other — the worker publishes the game before flipping the phase, and
            // `release` refuses to drop an unsaved one.) The old `NotReadable` here produced a
            // self-contradicting message: "is Cancelled; … can only be read once it has
            // finished or been cancelled".
            let path = saved_to.ok_or(JobError::NeverRan(id))?;
            let mut game = engine::load(&path, job.task.max_memory_bytes())?.0;
            // The file has no bunching field, so a game that was solved with it comes back
            // *without* it here — re-applying (EVs included, see `engine::reapply_bunching`)
            // is what keeps a released job's answers identical to its pre-release ones. Kept
            // out of `*guard` until it succeeds: publishing the bare game on failure would
            // answer the next read with silently different numbers, which is strictly worse
            // than this error.
            if let Task::Solve(spot) = &job.task {
                if let Some(bref) = &spot.bunching {
                    let wrap = |reason: String| JobError::BunchingUnavailable { id, reason };
                    let data = {
                        // The job's own retained Arc survives `release` and `forget` of the
                        // preparation; the shared resolution path is only the fallback.
                        let own = job.bunching_data.lock().unwrap().clone();
                        match own {
                            Some(data) => data,
                            None => resolve_bunching(&self.shared, bref, spot.max_memory_bytes)
                                .map_err(|e| wrap(e.to_string()))?,
                        }
                    };
                    engine::reapply_bunching(&mut game, &data).map_err(wrap)?;
                    *job.bunching_data.lock().unwrap() = Some(data);
                }
            }
            *guard = Some(game);
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
    ///
    /// "After the job it is on" is the library's half of the story; what happens to that job
    /// depends on who is embedding this. The binary exits right after answering the `shutdown`
    /// command, killing the detached worker mid-solve — an in-flight job is abandoned, and a
    /// `savePath` it was queued with is never written. Queued jobs simply never run, and no
    /// terminal frame is emitted for either kind. A host that wants the in-flight solve kept
    /// should `cancel` it and wait for its terminal frame before shutting down: a cancelled
    /// solve is still finalized and, if it asked for one, saved.
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

/// Drop one job's product — its game, or a bunching job's data — and say so. Assumes the caller
/// has established that it is recoverable. On a *solve* job that used bunching data, only the
/// game goes: the retained `Arc` is what lets the reload re-apply the effect.
///
/// The game (or data) guard is released before the status lock is taken. Only `node`'s reload
/// path nests them, in game → bunching_data → status order; nothing nests them the other way,
/// which is what keeps the module deadlock-free.
fn release_recoverable(shared: &Arc<Shared>, job: &Arc<Job>) {
    let dropped_game;
    let mut dropped_data = None;
    match &job.task {
        Task::Solve(_) => {
            dropped_game = job.game.lock().unwrap().take();
            if dropped_game.is_none() {
                // Already released. Nothing changed, so nothing is announced.
                return;
            }
        }
        Task::Bunching(_) => {
            dropped_game = None;
            dropped_data = job.bunching_data.lock().unwrap().take();
            if dropped_data.is_none() {
                return;
            }
        }
    }
    // Dropped outside both locks: freeing gigabytes is not instant and a `progress` query must not
    // queue behind it.
    drop(dropped_game);
    drop(dropped_data);

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
///
/// Bunching jobs are deliberately skipped: their 62 MB is two orders of magnitude below a tree,
/// and a queued solve is often about to need exactly that data — sweeping it would trade a
/// rounding error of memory for a 62 MB reload per solve. `release` still works on them
/// explicitly.
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
            status.kind == JobKind::Solve
                && status.phase.is_readable()
                && status.saved_to.is_some()
                && status.resident
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

    match &job.task {
        Task::Solve(spot) => run_solve_job(shared, job, spot, started),
        Task::Bunching(spec) => run_bunching_job(shared, job, spec, started),
    }
}

fn run_solve_job(shared: &Arc<Shared>, job: &Arc<Job>, spot: &Spot, started: Instant) {
    // Before allocating: hand back every tree that is already on disk. This is the difference
    // between an afternoon of solves costing one tree of memory and costing all of them.
    let id = job.status.lock().unwrap().job_id;
    release_others(shared, Some(id));

    // The reference is resolved fresh, here, rather than at submit: the preparation may not
    // even have run yet when the solve was queued (the FIFO guarantees it has by now), and a
    // submit-time Arc would strand data behind a solve cancelled while queued.
    let bunching = match &spot.bunching {
        None => None,
        Some(bref) => match resolve_bunching(shared, bref, spot.max_memory_bytes) {
            Ok(data) => Some(data),
            Err(e) => return fail(shared, job, started, e.to_string()),
        },
    };
    if let Some(data) = &bunching {
        // Retained for the job's whole life — `release` drops only the tree — so the reload
        // after a release can re-apply an effect the file format does not record.
        *job.bunching_data.lock().unwrap() = Some(Arc::clone(data));
    }

    // The engine panics rather than returning an error for several kinds of misuse, and it is
    // full of `unsafe` in its hot loops. A panic here must fail one job, not wedge the queue and
    // silence every future solve.
    let built = std::panic::catch_unwind(AssertUnwindSafe(|| {
        engine::build_with_bunching(spot, bunching.as_deref())
    }));

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
            &spot.stop,
            spot.pot,
            &job.cancel,
            |iterations, exploitability| {
                let mut status = job.status.lock().unwrap();
                status.iterations = iterations;
                status.exploitability = Some(exploitability);
                // The same decimation the engine loop applies to its own copy; the terminal
                // frame overwrites with the authoritative curve, so any drift between the two
                // decimation schedules ends with the job.
                engine::push_sample(
                    &mut status.history,
                    Sample {
                        iteration: iterations,
                        exploitability,
                    },
                );
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

    let saved_to = spot.save_path.as_deref().and_then(|path| {
        // A failed save must not turn a finished solve into a failure — the answer is in memory
        // and still worth having — so it is reported in `error` while the phase stays `Done`.
        match engine::save(
            &game,
            path,
            spot.memo.as_deref().unwrap_or_default(),
            spot.compression_level,
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

fn run_bunching_job(shared: &Arc<Shared>, job: &Arc<Job>, spec: &BunchingSpec, started: Instant) {
    // Rebuilt from the spec rather than kept from submit time — `new` allocates nothing, and a
    // submit-time instance would have stranded state behind a job cancelled while queued. The
    // same panic containment as the solve path: every engine call in this worker is inside a
    // catch_unwind so a bug fails one job, not the queue.
    let built = std::panic::catch_unwind(AssertUnwindSafe(|| engine::validate_bunching(spec)));
    let mut data = match built {
        Ok(Ok(data)) => data,
        Ok(Err(e)) => return fail(shared, job, started, e.to_string()),
        Err(payload) => return fail(shared, job, started, crate::describe_panic(&*payload)),
    };

    {
        let mut status = job.status.lock().unwrap();
        status.phase = Phase::Running;
        status.elapsed_ms = started.elapsed().as_millis() as u64;
        let snapshot = status.clone();
        drop(status);
        shared.emit_status(&snapshot);
    }

    let mut last_emit = Instant::now();
    let cancelled = std::panic::catch_unwind(AssertUnwindSafe(|| {
        for stage in 1u8..=3 {
            if job.cancel.load(Ordering::Relaxed) {
                return true;
            }
            // The engine's manual phase API: each `prepare` panics unless the previous phase
            // sits at exactly 100%, which this sequencing guarantees; the panics stay
            // theoretical, and the catch_unwind around us is what keeps them one-job-sized.
            match stage {
                1 => data.phase1_prepare(),
                2 => data.phase2_prepare(),
                _ => data.phase3_prepare(),
            }
            while data.progress_percent() < 100 {
                // Per percent-step, the same promptness class as the solve loop's
                // per-iteration check: a step is at worst seconds, on the four-player tables.
                if job.cancel.load(Ordering::Relaxed) {
                    return true;
                }
                match stage {
                    1 => data.phase1_proceed_by_percent(),
                    2 => data.phase2_proceed_by_percent(),
                    _ => data.phase3_proceed_by_percent(),
                }

                // The peak table is hand-derived from the engine's allocation schedule; if the
                // engine ever reschedules, say so in the log rather than failing a job that is
                // in fact succeeding.
                #[cfg(debug_assertions)]
                {
                    let claimed = engine::PEAK_BUNCHING_BYTES[data.fold_ranges().len() - 1];
                    if data.memory_usage() > claimed {
                        eprintln!(
                            "pkwiz-solver: bunching preparation live memory {} exceeds the \
                             claimed peak {claimed}; PEAK_BUNCHING_BYTES is stale",
                            data.memory_usage(),
                        );
                    }
                }

                let mut status = job.status.lock().unwrap();
                if let Some(bunching) = status.bunching.as_mut() {
                    bunching.stage = stage;
                    bunching.stage_percent = data.progress_percent();
                    bunching.overall_percent = (((u16::from(stage) - 1) * 100
                        + u16::from(data.progress_percent()))
                        / 3) as u8;
                }
                status.elapsed_ms = started.elapsed().as_millis() as u64;
                // Throttled, like solve progress: 300 steps in seconds would drown the pipe.
                let due = last_emit.elapsed() >= shared.throttle;
                let snapshot = due.then(|| status.clone());
                drop(status);
                if let Some(snapshot) = snapshot {
                    last_emit = Instant::now();
                    shared.emit_status(&snapshot);
                }
            }
        }
        false
    }));

    let cancelled = match cancelled {
        Ok(c) => c,
        Err(payload) => return fail(shared, job, started, crate::describe_panic(&*payload)),
    };

    if cancelled {
        // Unlike a cancelled solve — which still finalizes into a readable, merely-worse
        // strategy — a half-computed table is unusable, so nothing is published: `save` answers
        // `nothing_to_save` and a solve referencing this job `bunching_not_ready`.
        drop(data);
        let mut status = job.status.lock().unwrap();
        status.phase = Phase::Cancelled;
        status.stopped = Some(Stopped::Cancelled);
        status.elapsed_ms = started.elapsed().as_millis() as u64;
        // Emitted under the lock — see `Shared::emit_status`.
        shared.emit_status(&status);
        return;
    }

    let saved_to = spec.save_path.as_deref().and_then(|path| {
        // As with a solve's `savePath`: a failed save is reported in `error` while the phase
        // stays `Done`, because the data in memory is still worth having.
        match engine::save_bunching(
            &data,
            path,
            spec.memo.as_deref().unwrap_or_default(),
            spec.compression_level,
        ) {
            Ok(()) => Some(path.to_owned()),
            Err(e) => {
                job.status.lock().unwrap().error = Some(e.to_string());
                None
            }
        }
    });

    let memory_bytes = data.memory_usage();
    *job.bunching_data.lock().unwrap() = Some(Arc::new(data));

    let mut status = job.status.lock().unwrap();
    status.phase = Phase::Done;
    if let Some(bunching) = status.bunching.as_mut() {
        bunching.stage = 3;
        bunching.stage_percent = 100;
        bunching.overall_percent = 100;
        bunching.memory_bytes = Some(memory_bytes);
    }
    status.saved_to = saved_to;
    status.resident = true;
    status.elapsed_ms = started.elapsed().as_millis() as u64;
    // Emitted under the lock — see `Shared::emit_status`.
    shared.emit_status(&status);
}

/// Turn a solve's bunching reference into data, at the moment it is needed.
///
/// A job ref prefers the preparation's resident data (an `Arc` clone, free) and falls back to
/// its saved file; a file ref always loads. Loads honour the *solve's* memory cap — the
/// preparation's cap governed preparing, not this reload.
fn resolve_bunching(
    shared: &Arc<Shared>,
    bref: &BunchingRef,
    max_memory_bytes: Option<u64>,
) -> Result<Arc<BunchingData>, JobError> {
    match bref {
        BunchingRef::Job { job_id } => {
            let prep = shared
                .jobs
                .lock()
                .unwrap()
                .get(job_id)
                .cloned()
                .ok_or(JobError::NoSuchJob(*job_id))?;
            if !matches!(prep.task, Task::Bunching(_)) {
                return Err(JobError::NotBunching { id: *job_id });
            }
            if let Some(data) = prep.bunching_data.lock().unwrap().as_ref() {
                return Ok(Arc::clone(data));
            }
            let (phase, saved_to) = {
                let status = prep.status.lock().unwrap();
                (status.phase, status.saved_to.clone())
            };
            match saved_to {
                Some(path) => Ok(Arc::new(engine::load_bunching(&path, max_memory_bytes)?.0)),
                None => Err(JobError::BunchingNotReady { id: *job_id, phase }),
            }
        }
        BunchingRef::File { path } => {
            Ok(Arc::new(engine::load_bunching(path, max_memory_bytes)?.0))
        }
    }
}

/// The engine's (sorted) flop, rendered for [`BunchingStatus::flop`].
fn render_flop(flop: [postflop_solver::Card; 3]) -> Vec<String> {
    flop.iter()
        .map(|c| {
            crate::convert::from_engine_card(*c).map_or_else(|_| "??".to_owned(), |c| c.to_string())
        })
        .collect()
}

/// The first three board cards, sorted the way the engine sorts a flop, rendered for comparison.
fn sorted_flop(board: &[pkwiz_range::Card]) -> Vec<String> {
    let mut flop = [board[0], board[1], board[2]];
    flop.sort_unstable_by_key(|c| c.index());
    flop.iter().map(ToString::to_string).collect()
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
    ///
    /// **At a chance node the index is the dealt card's id (0–51), not a position in
    /// [`Self::actions`].** A host that walks the tree by enumerating `actions` and sending
    /// positions will be right at every decision node and wrong at exactly the chance nodes;
    /// parse the card out of the rendered `Chance(..)` entry instead.
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
    /// Expected value of each action of each of the acting player's hands, before the
    /// strategy is applied: `ev_detail[a][h]` is what hand `h` makes by taking action `a`
    /// (same units and perspective as [`Self::ev`], which is the strategy-weighted average of
    /// these rows). Rows are in [`Self::actions`] order, exactly like [`Self::strategy`].
    ///
    /// Two engine conventions surface here: a `Fold` row is exactly `0.0` for every hand (the
    /// engine's fold-EV convention, not a computed value), and a hand with zero normalized
    /// weight is `0.0` in every row. Empty at a terminal or chance node.
    #[serde(default)]
    pub ev_detail: Vec<Vec<f32>>,
    /// Whether any hand's strategy is pinned at this node (see `Spot::locks`). Always `false`
    /// at terminal and chance nodes. On a locked node, [`Self::strategy`] already shows the
    /// pinned frequencies — the engine applies the lock overlay at read time.
    #[serde(default)]
    pub is_locked: bool,
    /// Range-weighted averages of the two arrays above — the numbers a header line shows.
    ///
    /// `null` at a terminal or chance node, where there is no acting player to average over.
    /// (These were previously `f32::NAN` internally, which serde_json also writes as `null` —
    /// the wire shape is unchanged; the type now says so, and the struct can deserialize its
    /// own output.)
    pub average_equity: Option<f32>,
    pub average_ev: Option<f32>,
}

fn node_view(game: &mut PostFlopGame, history: &[usize]) -> Result<NodeView, JobError> {
    // The same validated replay `apply_locks` uses; at a chance node the "action" is the dealt
    // card id, not an offset into a list.
    engine::walk(game, history).map_err(|step| JobError::BadHistory {
        index: step.index,
        available: step.available,
    })?;

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
            ev_detail: Vec::new(),
            is_locked: false,
            average_equity: None,
            average_ev: None,
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
    // Chunked exactly like `strategy` above so the two matrices can never disagree on width.
    let ev_detail: Vec<Vec<f32>> = game
        .expected_values_detail(player)
        .chunks(num_hands.max(1))
        .map(<[f32]>::to_vec)
        .collect();
    let is_locked = game.current_locking_strategy().is_some();
    let average_equity = Some(postflop_solver::compute_average(&equity, &weights));
    let average_ev = Some(postflop_solver::compute_average(&ev, &weights));

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
        ev_detail,
        is_locked,
        average_equity,
        average_ev,
    })
}

fn render_action(action: &Action) -> String {
    crate::convert::action_to_string(action)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finish(jobs: &Jobs, id: JobId) -> JobStatus {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let status = jobs.status(id).expect("the job exists");
            if status.phase.is_terminal() {
                return status;
            }
            assert!(Instant::now() < deadline, "job {id} never finished");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn the_sweep_spares_a_resident_bunching_job() {
        let dir = std::env::temp_dir().join(format!("pkwiz-jobs-sweep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sweep.bin");

        let jobs = Jobs::new(Arc::new(crate::Silent));
        let board: Vec<pkwiz_range::Card> = ["2c", "7d", "Th", "4s", "Qd"]
            .iter()
            .map(|c| pkwiz_range::Card::parse(c).unwrap())
            .collect();
        let mut spot = Spot::from_hand(&board, 100, 100, "QQ+", "QQ+");
        spot.stop.max_iterations = 10;
        spot.save_path = Some(path.to_string_lossy().into_owned());
        let solved = finish(&jobs, jobs.submit(spot).unwrap().job_id);
        assert!(solved.resident && solved.saved_to.is_some());

        // A synthetic `done` bunching row: the sweep reads only kind, phase, savedTo and
        // resident, so cheap unprocessed data stands in for a real 62 MB preparation.
        let spec: BunchingSpec =
            serde_json::from_str(r#"{"foldRanges":["AA"],"flop":"2c7dTh"}"#).unwrap();
        let data = spec.validate().unwrap();
        let id = jobs.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let mut status = JobStatus::new_bunching(id, 1, render_flop(data.flop()));
        status.phase = Phase::Done;
        status.saved_to = Some("/nowhere.bunching".to_owned());
        status.resident = true;
        let job = Arc::new(Job {
            task: Task::Bunching(spec),
            cancel: AtomicBool::new(false),
            status: Mutex::new(status),
            game: Mutex::new(None),
            bunching_data: Mutex::new(Some(Arc::new(data))),
        });
        jobs.shared.jobs.lock().unwrap().insert(id, job);

        release_others(&jobs.shared, None);

        assert!(
            !jobs.status(solved.job_id).unwrap().resident,
            "the saved solve's tree goes back"
        );
        assert!(
            jobs.status(id).unwrap().resident,
            "62 MB of about-to-be-reused data is not what the sweep exists to reclaim"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
