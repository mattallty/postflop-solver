//! Does the solver actually solve?
//!
//! Three things are worth asserting about a CFR implementation you did not write, and none of
//! them need a reference solver:
//!
//! 1. **Exploitability falls.** That is the entire claim of the algorithm. If it does not fall,
//!    nothing else about the output means anything.
//! 2. **Symmetry is respected.** Spots exist whose answer is forced by the shape of the problem —
//!    identical ranges must have identical equity; a board that makes every hand a chop must
//!    produce zero EV for both players. Those are checkable exactly.
//! 3. **Stopping stops.** A cancel that only takes effect when the solve would have finished
//!    anyway is not a cancel.
//!
//! Everything here is deliberately tiny: river trees over nine to eighteen combinations, tens of
//! iterations. The engine's own suite already checks its numbers against PioSOLVER (
//! four `--ignored` tests, 357 s); repeating that in CI would buy nothing and cost minutes.

use std::sync::Arc;
use std::time::{Duration, Instant};

use pkwiz_solver::spot::{BoardSpec, RangeSpec, Sizing, StreetSizing};
use pkwiz_solver::{
    Collector, Emit, JobError, JobStatus, Jobs, Phase, Silent, Spot, Stop, Stopped,
};

/// A river spot: no chance nodes, so the tree is as small as a real spot gets.
fn river(board: &str, range: &str, iterations: u32) -> Spot {
    Spot {
        oop: RangeSpec::Notation(range.to_owned()),
        ip: RangeSpec::Notation(range.to_owned()),
        board: BoardSpec::Text(board.to_owned()),
        pot: 100,
        effective_stack: 100,
        sizing: Sizing {
            flop: StreetSizing::none(),
            turn: StreetSizing::none(),
            river: StreetSizing {
                bet: "50%".to_owned(),
                raise: "2.5x".to_owned(),
            },
            ..Sizing::default()
        },
        rake: pkwiz_solver::spot::Rake::default(),
        stop: Stop {
            max_iterations: iterations,
            // Zero, so the iteration cap is what stops it and the test controls the runtime
            // exactly rather than depending on how fast this spot happens to converge.
            target_exploitability: Some(0.0),
            target_exploitability_pct: 0.0,
            check_interval: 5,
        },
        compress: false,
        max_memory_bytes: None,
        save_path: None,
        compression_level: Spot::default_compression_level(),
        memo: None,
        added_lines: Vec::new(),
        removed_lines: Vec::new(),
        locks: Vec::new(),
        bunching: None,
    }
}

/// Block until the job is terminal, or fail.
fn finish(jobs: &Jobs, id: u64, within: Duration) -> JobStatus {
    let deadline = Instant::now() + within;
    loop {
        let status = jobs.status(id).expect("the job exists");
        if status.phase.is_terminal() {
            return status;
        }
        assert!(
            Instant::now() < deadline,
            "job {id} still {:?} after {within:?}",
            status.phase
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// A directory of this test's own, so the file-shaped tests can run alongside each other and clean
/// up after themselves without deleting anyone else's fixtures.
fn temp_dir(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("pkwiz-solver-{}-{label}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn solve(spot: Spot) -> (Jobs, JobStatus) {
    let jobs = Jobs::new(Arc::new(Silent));
    let queued = jobs.submit(spot).expect("the spot is valid");
    let done = finish(&jobs, queued.job_id, Duration::from_secs(60));
    (jobs, done)
}

#[test]
fn exploitability_falls_which_is_the_whole_claim_of_cfr() {
    let (_jobs, done) = solve(river("2c7dTh4sQd", "QQ+,AKs,76s", 200));

    assert_eq!(done.phase, Phase::Done);
    assert_eq!(done.stopped, Some(Stopped::IterationCap));
    assert_eq!(done.iterations, 200);

    let history = &done.history;
    assert!(history.len() > 10, "expected a curve, got {history:?}");
    let first = history[0].exploitability;
    let last = history.last().unwrap().exploitability;

    // Two orders of magnitude over two hundred iterations on a tree this small. A broken solver
    // fails this by miles; a merely slow one still passes, which is the right sensitivity.
    assert!(
        last < first / 100.0,
        "exploitability went {first} -> {last}"
    );

    // Monotone, but at the right resolution. Discounted CFR is *not* monotone iteration to
    // iteration — it resets its cumulative strategy whenever the iteration count reaches a power
    // of four, and the measurement right after a reset routinely jumps by 4×. Asserting
    // `history[i+1] <= history[i]` would therefore assert something false about a working solver.
    //
    // What is true, and what convergence actually means, is that each stretch of the run is
    // decisively better than the last. Quartered, this run goes 8.0 -> 0.84 -> 0.23 -> 0.098.
    let quarter = history.len() / 4;
    let means: Vec<f32> = (0..4)
        .map(|q| {
            let slice = &history[q * quarter..(q + 1) * quarter];
            slice.iter().map(|s| s.exploitability).sum::<f32>() / slice.len() as f32
        })
        .collect();
    for pair in means.windows(2) {
        assert!(
            pair[1] < pair[0] / 2.0,
            "exploitability stalled across the run: {means:?} from {history:?}"
        );
    }

    // Zero-sum without rake, which is a free arithmetic check on the finalized values.
    let ev = done.ev.expect("a finished job has EVs");
    assert!((ev[0] + ev[1]).abs() < 0.01, "{ev:?} is not zero-sum");
}

#[test]
fn a_tree_with_no_betting_is_solved_before_it_starts() {
    // The strongest forced answer available: if neither player may bet, the only line is
    // check-check, so no strategy exists to exploit and exploitability is exactly zero. With
    // identical ranges the spot is also symmetric, so each player's EV is exactly half the pot —
    // reported as zero, because `compute_current_ev` subtracts that bias.
    let mut spot = river("2c7dTh4sQd", "QQ+", 50);
    spot.sizing.river = StreetSizing::none();
    spot.sizing.add_allin_threshold = 0.0;
    spot.sizing.force_allin_threshold = 0.0;

    let (jobs, done) = solve(spot);
    assert_eq!(done.phase, Phase::Done);
    // Zero up to f32 rounding on a sum over nine hands; the engine reports -7e-7 here.
    assert!(
        done.exploitability.unwrap().abs() < 1e-4,
        "a game with no decisions cannot be exploited, got {:?}",
        done.exploitability
    );

    let ev = done.ev.unwrap();
    assert!(ev[0].abs() < 1e-3 && ev[1].abs() < 1e-3, "{ev:?}");

    // And the two players hold the same equity, because they hold the same range.
    let oop = jobs.node(done.job_id, &[]).unwrap();
    assert_eq!(oop.player, Some(0));
    assert!(
        (oop.average_equity.unwrap() - 0.5).abs() < 1e-4,
        "{}",
        oop.average_equity.unwrap()
    );
    // The whole tree is one action for each player.
    assert_eq!(oop.actions, ["Check"]);
    let ip = jobs.node(done.job_id, &[0]).unwrap();
    assert_eq!(ip.player, Some(1));
    assert!((ip.average_equity.unwrap() - 0.5).abs() < 1e-4);
}

#[test]
fn a_board_that_chops_every_hand_forces_the_answer() {
    // A royal flush on the board: every hand plays the board, so all equity is 50% no matter what
    // anyone holds, and folding is strictly dominated — it turns a guaranteed chop into a
    // guaranteed loss of half the pot. Nothing about that depends on a reference solver.
    let (jobs, done) = solve(river("AsKsQsJsTs", "QQ+", 300));
    assert_eq!(done.phase, Phase::Done);

    let ev = done.ev.unwrap();
    assert!(
        ev[0].abs() < 0.5 && ev[1].abs() < 0.5,
        "nobody can profit from a chopped board, got {ev:?}"
    );

    let root = jobs.node(done.job_id, &[]).unwrap();
    assert!((root.average_equity.unwrap() - 0.5).abs() < 1e-4);
    // Every single hand, not just the average.
    assert!(
        root.equity.iter().all(|e| (e - 0.5).abs() < 1e-4),
        "some hand beats a royal flush: {:?}",
        root.equity
    );

    // Facing a bet, folding a chop is giving away half the pot, so the equilibrium folds ~never.
    let bet = root
        .actions
        .iter()
        .position(|a| a.starts_with("Bet") || a.starts_with("AllIn"))
        .expect("the river tree offers a bet");
    let facing = jobs.node(done.job_id, &[bet]).unwrap();
    assert_eq!(facing.player, Some(1));
    let fold = facing
        .actions
        .iter()
        .position(|a| a == "Fold")
        .expect("facing a bet, folding is an option");
    let worst = facing.strategy[fold]
        .iter()
        .fold(0.0f32, |acc, f| acc.max(*f));
    assert!(
        worst < 0.05,
        "some hand folds a guaranteed chop {worst} of the time"
    );
}

#[test]
fn cancel_stops_the_work_rather_than_the_reporting() {
    // A million iterations on a tiny tree: minutes of work, so a cancel that only took effect at
    // the end would be unmissable here.
    let mut spot = river("2c7dTh4sQd", "QQ+,AKs,76s", 1_000_000);
    spot.stop.check_interval = 20;

    let jobs = Jobs::new(Arc::new(Silent));
    let queued = jobs.submit(spot).expect("valid");
    let id = queued.job_id;

    // Wait until it is genuinely iterating, so we are cancelling work and not a queue entry.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let status = jobs.status(id).unwrap();
        if status.phase == Phase::Running && status.iterations >= 20 {
            break;
        }
        assert!(Instant::now() < deadline, "never started: {status:?}");
        std::thread::sleep(Duration::from_millis(2));
    }

    let asked = Instant::now();
    let after = jobs.cancel(id).unwrap();
    let ack = asked.elapsed();
    assert!(
        ack < Duration::from_millis(100),
        "cancel blocked for {ack:?}; it must not wait for the solver"
    );
    assert!(!after.phase.is_terminal(), "it was still running");

    let done = finish(&jobs, id, Duration::from_secs(5));
    let stopped = asked.elapsed();
    assert_eq!(done.phase, Phase::Cancelled);
    assert_eq!(done.stopped, Some(Stopped::Cancelled));
    assert!(
        stopped < Duration::from_secs(5),
        "took {stopped:?} to actually stop"
    );
    assert!(
        done.iterations < 100_000,
        "ran {} of a million iterations, which is not cancelling",
        done.iterations
    );

    // Cancelled early, but still finalized: the point of a stop button is to keep what you have.
    let root = jobs.node(id, &[]).unwrap();
    assert!(!root.strategy.is_empty());
    assert!(root.strategy[0].len() == root.hands.len());
}

#[test]
fn cancelling_a_queued_job_never_starts_it() {
    let jobs = Jobs::new(Arc::new(Silent));
    // The first job occupies the single worker; the second is cancelled while still queued.
    let first = jobs
        .submit(river("2c7dTh4sQd", "QQ+,AKs,76s", 200_000))
        .unwrap();
    let second = jobs.submit(river("2c7dTh4sQd", "QQ+", 200_000)).unwrap();
    assert_eq!(second.phase, Phase::Queued);

    let cancelled = jobs.cancel(second.job_id).unwrap();
    assert_eq!(cancelled.phase, Phase::Cancelled);
    assert_eq!(cancelled.iterations, 0);

    jobs.cancel(first.job_id).unwrap();
    finish(&jobs, first.job_id, Duration::from_secs(10));
    // Still zero after the worker has drained the queue: it was skipped, not run.
    assert_eq!(jobs.status(second.job_id).unwrap().iterations, 0);
    assert_eq!(jobs.list().len(), 2);
}

#[test]
fn progress_is_pushed_while_the_solve_runs_not_only_at_the_end() {
    let collector = Arc::new(Collector::default());
    let jobs = Jobs::with_throttle(
        Arc::clone(&collector) as Arc<dyn pkwiz_solver::Emit>,
        Duration::ZERO,
    );
    let id = jobs
        .submit(river("2c7dTh4sQd", "QQ+,AKs,76s", 400))
        .unwrap()
        .job_id;
    finish(&jobs, id, Duration::from_secs(60));

    let events = collector.events();
    assert!(events.len() > 5, "only {} frames pushed", events.len());
    assert_eq!(events[0]["job"]["phase"], "queued");
    assert_eq!(events.last().unwrap()["job"]["phase"], "done");

    // Iterations climb across the frames, which is the property a progress bar needs and that
    // a single terminal frame would satisfy vacuously.
    let iterations: Vec<u64> = events
        .iter()
        .map(|e| e["job"]["iterations"].as_u64().unwrap())
        .collect();
    assert!(
        iterations.windows(2).all(|w| w[0] <= w[1]),
        "{iterations:?}"
    );
    assert!(
        iterations.iter().any(|i| *i > 0 && *i < 400),
        "no frame arrived mid-solve: {iterations:?}"
    );
    // Every frame is a complete status, not a delta.
    assert!(events
        .iter()
        .all(|e| e["job"]["jobId"] == id && e["job"]["startingPot"] == 100));
}

#[test]
fn a_solution_survives_a_round_trip_through_a_file() {
    let dir = temp_dir("round-trip");
    let path = dir.join("round-trip.bin");

    let mut spot = river("2c7dTh4sQd", "QQ+,AKs", 40);
    spot.save_path = Some(path.to_string_lossy().into_owned());
    spot.memo = Some("hand #4471, river".to_owned());

    let (jobs, done) = solve(spot);
    assert_eq!(done.saved_to.as_deref(), Some(&*path.to_string_lossy()));
    assert!(done.error.is_none(), "{:?}", done.error);
    let before = jobs.node(done.job_id, &[]).unwrap();

    let reopened = jobs.open(&path.to_string_lossy(), None, None).unwrap();
    assert_eq!(reopened.phase, Phase::Done);
    assert_ne!(reopened.job_id, done.job_id);
    let after = jobs.node(reopened.job_id, &[]).unwrap();

    assert_eq!(after.hands, before.hands);
    assert_eq!(after.strategy, before.strategy);
    assert_eq!(after.equity, before.equity);
    // The engine sorts the flop (it derives suit isomorphism from that order) and leaves the turn
    // and river where they are, so `2c7dTh4sQd` reads back in exactly this order.
    assert_eq!(after.board, ["2c", "7d", "Th", "4s", "Qd"]);

    std::fs::remove_file(&path).ok();
}

#[test]
fn a_saved_solution_is_compressed_and_the_raw_form_still_opens() {
    let dir = temp_dir("compression");
    let compressed = dir.join("compressed.bin");
    let raw = dir.join("raw.bin");

    let mut spot = river("2c7dTh4sQd", "22+,A2s+", 20);
    spot.save_path = Some(compressed.to_string_lossy().into_owned());

    let (jobs, done) = solve(spot);
    assert!(done.error.is_none(), "{:?}", done.error);
    let before = jobs.node(done.job_id, &[]).unwrap();

    // The same tree written without compression, for something to compare against.
    pkwiz_solver::engine::save(
        &pkwiz_solver::engine::load(&compressed.to_string_lossy(), None)
            .unwrap()
            .0,
        &raw.to_string_lossy(),
        "",
        None,
    )
    .unwrap();

    let small = std::fs::metadata(&compressed).unwrap().len();
    let large = std::fs::metadata(&raw).unwrap().len();
    assert!(
        small < large,
        "compressed {small} is not smaller than raw {large}"
    );

    // Both forms open, and to the same strategy. The uncompressed one is the shape every file
    // written before this was turned on has, so this is also the back-compatibility check.
    for path in [&compressed, &raw] {
        let reopened = jobs.open(&path.to_string_lossy(), None, None).unwrap();
        let after = jobs.node(reopened.job_id, &[]).unwrap();
        assert_eq!(after.strategy, before.strategy, "{}", path.display());
        assert_eq!(after.equity, before.equity, "{}", path.display());
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn solution_files_carry_the_format_version_we_claim() {
    // `protocol::ENGINE_FORMAT` duplicates a string the engine keeps private and refuses to decode
    // a mismatch of, so `ENGINE_COMPATIBLE_REVS` is only as good as that duplicate being right.
    // The engine stamps it into the body of every file it writes, which makes the claim checkable:
    // written raw, the bytes are there to be found.
    let dir = temp_dir("format");
    let path = dir.join("stamped.bin");

    let mut spot = river("2c7dTh4sQd", "QQ+", 10);
    spot.compression_level = None;
    spot.save_path = Some(path.to_string_lossy().into_owned());

    let (_jobs, done) = solve(spot);
    assert!(done.error.is_none(), "{:?}", done.error);

    let bytes = std::fs::read(&path).unwrap();
    let stamp = pkwiz_solver::protocol::ENGINE_FORMAT.as_bytes();
    assert!(
        bytes.windows(stamp.len()).any(|w| w == stamp),
        "no {:?} in the file the engine just wrote — ENGINE_FORMAT has drifted from the engine",
        pkwiz_solver::protocol::ENGINE_FORMAT
    );

    // And the revision we claim compatibility with is a superset of the one we are built on.
    assert!(pkwiz_solver::protocol::ENGINE_COMPATIBLE_REVS
        .contains(&pkwiz_solver::protocol::ENGINE_REV));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn releasing_a_saved_job_hands_back_its_tree_and_reading_it_reloads() {
    let dir = temp_dir("release");
    let path = dir.join("released.bin");

    let mut spot = river("2c7dTh4sQd", "QQ+,AKs", 40);
    spot.save_path = Some(path.to_string_lossy().into_owned());

    let (jobs, done) = solve(spot);
    assert!(done.resident, "a job that just solved holds its tree");
    let before = jobs.node(done.job_id, &[]).unwrap();

    let released = jobs.release(done.job_id).unwrap();
    assert!(!released.resident);
    // Everything else about the job is exactly as it was: releasing is not forgetting.
    assert_eq!(released.phase, done.phase);
    assert_eq!(released.exploitability, done.exploitability);
    assert_eq!(released.ev, done.ev);
    assert_eq!(released.saved_to, done.saved_to);
    assert_eq!(released.history, done.history);

    // Releasing twice is a no-op, not an error.
    assert!(!jobs.release(done.job_id).unwrap().resident);

    // And the strategy is still readable, identically, because `node` reloads the file.
    let after = jobs.node(done.job_id, &[]).unwrap();
    assert_eq!(after.strategy, before.strategy);
    assert_eq!(after.equity, before.equity);
    assert_eq!(after.ev, before.ev);
    assert!(
        jobs.status(done.job_id).unwrap().resident,
        "reading a released job puts it back in memory"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_job_with_no_file_behind_it_refuses_to_be_released() {
    // The in-memory tree is the only copy, so dropping it would silently throw the solve away.
    let (jobs, done) = solve(river("2c7dTh4sQd", "QQ+", 20));
    assert!(done.saved_to.is_none());

    let refused = jobs.release(done.job_id).unwrap_err();
    assert_eq!(refused.code(), "not_recoverable");
    assert!(
        jobs.status(done.job_id).unwrap().resident,
        "a refused release must not have released anything"
    );
    assert!(jobs.node(done.job_id, &[]).is_ok());
}

#[test]
fn starting_a_solve_hands_back_the_trees_that_are_already_on_disk() {
    // The reason any of this exists: an afternoon of solves should cost one tree of memory, not
    // one per solve. A second solve is what triggers the release of the first.
    let dir = temp_dir("release-others");
    let saved_path = dir.join("saved.bin");

    let jobs = Jobs::new(Arc::new(Silent));

    let mut saved = river("2c7dTh4sQd", "QQ+,AKs", 30);
    saved.save_path = Some(saved_path.to_string_lossy().into_owned());
    let first = finish(
        &jobs,
        jobs.submit(saved).unwrap().job_id,
        Duration::from_secs(60),
    );

    assert!(
        first.resident,
        "nothing has happened since, so the first solve still holds its tree"
    );

    // A second job, with nowhere to write: starting it releases the first, and it must itself
    // survive every later sweep, because the tree in memory is the only copy of its answer.
    let unsaved = finish(
        &jobs,
        jobs.submit(river("AsKsQsJsTs", "QQ+", 20)).unwrap().job_id,
        Duration::from_secs(60),
    );

    assert!(
        !jobs.status(first.job_id).unwrap().resident,
        "the saved job's tree should have been handed back when the next solve started"
    );
    assert!(jobs.status(unsaved.job_id).unwrap().resident);

    let third = finish(
        &jobs,
        jobs.submit(river("2c7dTh4sQd", "KK+", 20)).unwrap().job_id,
        Duration::from_secs(60),
    );

    assert!(
        jobs.status(unsaved.job_id).unwrap().resident,
        "the unsaved job is the only copy of its answer and must be left alone"
    );
    assert!(
        jobs.status(third.job_id).unwrap().resident,
        "the job that just solved keeps its own tree"
    );

    // And all three answer, the released one by reloading its file.
    assert!(jobs.node(first.job_id, &[]).is_ok());
    assert!(jobs.node(unsaved.job_id, &[]).is_ok());
    assert!(jobs.node(third.job_id, &[]).is_ok());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn opening_a_file_hands_back_the_trees_that_are_already_on_disk() {
    // The browsing path, which never calls `solve`: a library walker only opens files. Sweeping on
    // `solve` alone would leave this accumulating one tree per open — including opening the *same*
    // file twice, which is two resident copies of one solution.
    let dir = temp_dir("release-on-open");
    let path = dir.join("browsed.bin");

    let jobs = Jobs::new(Arc::new(Silent));

    let mut saved = river("2c7dTh4sQd", "QQ+,AKs", 30);
    saved.save_path = Some(path.to_string_lossy().into_owned());
    let solved = finish(
        &jobs,
        jobs.submit(saved).unwrap().job_id,
        Duration::from_secs(60),
    );
    assert!(solved.resident);
    let before = jobs.node(solved.job_id, &[]).unwrap();

    // First open: the solve's own tree goes back.
    let first = jobs.open(&path.to_string_lossy(), None, None).unwrap();
    assert!(first.resident);
    assert!(
        !jobs.status(solved.job_id).unwrap().resident,
        "opening a file should have handed back the solved job's tree"
    );

    // Second open of the same file: the first opened job goes back too, rather than the session
    // holding two copies of one solution.
    let second = jobs.open(&path.to_string_lossy(), None, None).unwrap();
    assert_ne!(second.job_id, first.job_id);
    assert!(
        !jobs.status(first.job_id).unwrap().resident,
        "a second open should have handed back the first opened tree"
    );
    assert!(jobs.status(second.job_id).unwrap().resident);

    // All three still answer, and with the same strategy.
    for id in [solved.job_id, first.job_id, second.job_id] {
        let view = jobs.node(id, &[]).unwrap();
        assert_eq!(view.strategy, before.strategy, "job {id}");
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_tree_too_big_for_its_budget_is_refused_with_a_number() {
    let mut spot = river("2c7dTh4sQd", "22+,A2s+,K2s+", 10);
    spot.max_memory_bytes = Some(1024);

    let jobs = Jobs::new(Arc::new(Silent));
    let id = jobs.submit(spot.clone()).unwrap().job_id;
    let done = finish(&jobs, id, Duration::from_secs(30));

    assert_eq!(done.phase, Phase::Failed);
    let message = done.error.expect("a failure says why");
    assert!(message.contains("maxMemoryBytes"), "{message}");
    assert!(message.contains("1024"), "{message}");

    // And the estimate answers the same question without allocating or failing.
    let estimate = pkwiz_solver::engine::estimate(&spot).unwrap();
    assert!(estimate.uncompressed > 1024);
    assert!(estimate.compressed < estimate.uncompressed);
}

#[test]
fn node_queries_are_refused_until_there_is_something_to_read() {
    let jobs = Jobs::new(Arc::new(Silent));
    let id = jobs
        .submit(river("2c7dTh4sQd", "QQ+", 1_000_000))
        .unwrap()
        .job_id;

    let err = jobs.node(id, &[]).unwrap_err();
    assert_eq!(err.code(), "not_readable");
    assert!(jobs.node(9999, &[]).unwrap_err().code() == "no_such_job");

    // Cancel only once it is genuinely iterating: a job cancelled while still queued never gets a
    // game at all, and would answer `not_readable` for the rest of its life — correctly, but that
    // is a different assertion than the one this test is making.
    let deadline = Instant::now() + Duration::from_secs(10);
    while jobs.status(id).unwrap().iterations == 0 {
        assert!(Instant::now() < deadline, "never started");
        std::thread::sleep(Duration::from_millis(2));
    }
    jobs.cancel(id).unwrap();
    finish(&jobs, id, Duration::from_secs(10));

    // And a history that does not describe a node is a reportable error, not a panic.
    let err = jobs.node(id, &[99]).unwrap_err();
    assert_eq!(err.code(), "bad_node");
    assert!(err.to_string().contains("out of range"), "{err}");
}

/// Realistic flop solves, for the record rather than for CI.
///
/// `cargo test -p pkwiz-solver --release -- --ignored --nocapture` prints how long a spot of the
/// size someone would actually ask for takes, and to what exploitability. The ranges and the
/// board are `postflop-solver`'s own `basic.rs` example, so the numbers are comparable to the
/// engine's published ones; the two sizings are our default and that example's richer tree.
#[test]
#[ignore = "tens of seconds to minutes — this is the benchmark, not a check"]
fn a_realistic_flop_solve() {
    let base = Spot {
        oop: RangeSpec::Notation(
            "66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s".to_owned(),
        ),
        ip: RangeSpec::Notation(
            "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+".to_owned(),
        ),
        board: BoardSpec::Text("Td9d6h".to_owned()),
        pot: 200,
        effective_stack: 900,
        sizing: Sizing::default(),
        rake: pkwiz_solver::spot::Rake::default(),
        stop: Stop {
            max_iterations: 1000,
            target_exploitability: None,
            // The industry convention: 0.5% of the pot.
            target_exploitability_pct: 0.5,
            check_interval: 10,
        },
        compress: false,
        max_memory_bytes: Some(32 * 1024 * 1024 * 1024),
        save_path: None,
        compression_level: Spot::default_compression_level(),
        memo: None,
        added_lines: Vec::new(),
        removed_lines: Vec::new(),
        locks: Vec::new(),
        bunching: None,
    };

    for (label, sizing) in [
        ("default (50% bet, 2.5x raise)", Sizing::default()),
        (
            "rich (60%, geometric, all-in; 2.5x raise)",
            Sizing {
                flop: StreetSizing {
                    bet: "60%, e, a".to_owned(),
                    raise: "2.5x".to_owned(),
                },
                turn: StreetSizing {
                    bet: "60%, e, a".to_owned(),
                    raise: "2.5x".to_owned(),
                },
                river: StreetSizing {
                    bet: "60%, e, a".to_owned(),
                    raise: "2.5x".to_owned(),
                },
                ..Sizing::default()
            },
        ),
    ] {
        let spot = Spot {
            sizing,
            ..base.clone()
        };
        let estimate = pkwiz_solver::engine::estimate(&spot).unwrap();
        let started = Instant::now();
        let (_jobs, done) = solve(spot);
        println!(
            "{label}: {:.2} GB ({:.2} GB compressed) | {:?} after {} iterations in {:.1}s | \
             exploitability {:.4} vs target {:.4} | ev {:?}",
            estimate.uncompressed as f64 / 1e9,
            estimate.compressed as f64 / 1e9,
            done.stopped,
            done.iterations,
            started.elapsed().as_secs_f64(),
            done.exploitability.unwrap(),
            done.target_exploitability,
            done.ev.unwrap(),
        );
        assert_eq!(done.phase, Phase::Done);
    }
}

#[test]
fn forget_discards_an_unsaved_job_and_its_row() {
    let (jobs, done) = solve(river("2c7dTh4sQd", "QQ+", 30));
    assert_eq!(done.phase, Phase::Done);
    assert!(done.saved_to.is_none());

    // `release` refuses an unsaved job — dropping its tree would discard the only copy — which
    // without `forget` left every unsaved solve resident until the process exited.
    assert_eq!(
        jobs.release(done.job_id),
        Err(JobError::NotRecoverable(done.job_id))
    );

    // `forget` is the deliberate discard: the response carries the final status, and then both
    // the tree and the row are gone.
    let last = jobs.forget(done.job_id).unwrap();
    assert_eq!(last.phase, Phase::Done);
    assert!(jobs.list().iter().all(|j| j.job_id != done.job_id));
    assert_eq!(
        jobs.status(done.job_id),
        Err(JobError::NoSuchJob(done.job_id))
    );
    assert_eq!(
        jobs.forget(done.job_id),
        Err(JobError::NoSuchJob(done.job_id))
    );
}

#[test]
fn a_job_that_is_not_finished_refuses_to_be_forgotten() {
    let mut spot = river("2c7dTh4sQd", "QQ+,AKs,76s", 1_000_000);
    spot.stop.check_interval = 20;

    let jobs = Jobs::new(Arc::new(Silent));
    let id = jobs.submit(spot).expect("valid").job_id;

    // While it is queued or running, `forget` must refuse: the worker would otherwise be left
    // solving into a row that no longer exists.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let status = jobs.status(id).unwrap();
        if status.phase == Phase::Running {
            break;
        }
        assert!(Instant::now() < deadline, "never started: {status:?}");
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(matches!(jobs.forget(id), Err(JobError::NotFinished { .. })));

    // Once terminal — here via cancel — it can go.
    jobs.cancel(id).unwrap();
    finish(&jobs, id, Duration::from_secs(10));
    assert!(jobs.forget(id).is_ok());
    assert_eq!(jobs.status(id), Err(JobError::NoSuchJob(id)));
}

#[test]
fn open_refuses_a_file_bigger_than_its_budget() {
    let dir = temp_dir("open-cap");
    let path = dir.join("cap.bin");

    let mut spot = river("2c7dTh4sQd", "QQ+", 30);
    spot.save_path = Some(path.to_string_lossy().into_owned());
    let (jobs, done) = solve(spot);
    assert!(done.saved_to.is_some());

    // A one-byte budget refuses any real tree — the same refusal-over-OOM-kill contract the
    // solve path provides — while the default budget still opens it.
    assert!(jobs.open(&path.to_string_lossy(), Some(1), None).is_err());
    assert!(jobs.open(&path.to_string_lossy(), None, None).is_ok());

    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_terminal_frame_is_final_even_when_cancel_races_the_worker() {
    // Cancel immediately after submitting, twenty times over: sometimes the cancel lands before
    // the worker claims the job, sometimes after. Whichever way each race falls, no job may emit
    // any frame after its terminal one — the bug this pins down had the worker hauling a job
    // already announced as `cancelled` back to `building` and solving it anyway.
    let collector = Arc::new(Collector::default());
    let jobs = Jobs::with_throttle(Arc::clone(&collector) as Arc<dyn Emit>, Duration::ZERO);

    let mut ids = Vec::new();
    for _ in 0..20 {
        let queued = jobs.submit(river("2c7dTh4sQd", "QQ+", 5)).unwrap();
        jobs.cancel(queued.job_id).unwrap();
        ids.push(queued.job_id);
    }
    for id in &ids {
        finish(&jobs, *id, Duration::from_secs(60));
    }

    let mut terminal = std::collections::HashSet::new();
    for event in collector.events() {
        let job = &event["job"];
        let id = job["jobId"].as_u64().unwrap();
        let phase = job["phase"].as_str().unwrap();
        assert!(
            !terminal.contains(&id),
            "job {id} emitted a `{phase}` frame after its terminal frame"
        );
        if matches!(phase, "done" | "cancelled" | "failed") {
            terminal.insert(id);
        }
    }
    // Every job ended exactly once.
    assert_eq!(terminal.len(), ids.len());
}

#[test]
fn a_job_cancelled_while_queued_says_so_when_read() {
    let jobs = Jobs::new(Arc::new(Silent));
    // The first job occupies the single worker; the second is cancelled while still queued, so
    // it never produces a game or a file.
    let first = jobs
        .submit(river("2c7dTh4sQd", "QQ+,AKs,76s", 200_000))
        .unwrap();
    let second = jobs.submit(river("2c7dTh4sQd", "QQ+", 200_000)).unwrap();
    jobs.cancel(second.job_id).unwrap();

    // Not the old self-contradiction ("is Cancelled; … can only be read once it has finished
    // or been cancelled") — a distinct answer a host can branch on.
    let err = jobs.node(second.job_id, &[]).unwrap_err();
    assert_eq!(err, JobError::NeverRan(second.job_id));
    assert_eq!(err.code(), "never_ran");

    jobs.cancel(first.job_id).unwrap();
    finish(&jobs, first.job_id, Duration::from_secs(10));
}

#[test]
fn a_lock_pins_the_strategy_and_the_node_says_so() {
    // Pin OOP's whole root range to pure Check; the solver must leave it there, optimize IP
    // against it, and the node must report both the pinned frequencies and the lock itself.
    let mut spot = river("2c7dTh4sQd", "QQ+,AKs,76s", 60);
    // Learn the root shape first: actions and hand count.
    let (jobs, done) = solve(spot.clone());
    let root = jobs.node(done.job_id, &[]).unwrap();
    let num_actions = root.actions.len();
    let num_hands = root.hands.len();
    let check = root
        .actions
        .iter()
        .position(|a| a == "Check")
        .expect("the root offers Check");
    assert!(!root.is_locked);

    let mut strategy = vec![vec![0.0f32; num_hands]; num_actions];
    strategy[check] = vec![1.0; num_hands];
    spot.locks = vec![pkwiz_solver::Lock {
        history: Vec::new(),
        strategy,
        hands: Some(root.hands.clone()),
    }];

    let (jobs, done) = solve(spot);
    assert_eq!(done.phase, Phase::Done, "{:?}", done.error);
    let locked = jobs.node(done.job_id, &[]).unwrap();
    assert!(locked.is_locked);
    for h in 0..num_hands {
        assert!(
            (locked.strategy[check][h] - 1.0).abs() < 1e-6,
            "hand {h} was not pinned: {:?}",
            locked.strategy
        );
    }
    // Deeper nodes are not locked.
    let after_check = jobs.node(done.job_id, &[check]).unwrap();
    assert!(!after_check.is_locked);
}

#[test]
fn a_structurally_bad_lock_fails_the_job_with_its_name() {
    // Wrong dimensions can only be discovered against the built tree, so the job fails.
    let mut spot = river("2c7dTh4sQd", "QQ+", 30);
    spot.locks = vec![pkwiz_solver::Lock {
        history: Vec::new(),
        strategy: vec![vec![1.0]], // 1x1 where the root needs actions x hands
        hands: None,
    }];
    let jobs = Jobs::new(Arc::new(Silent));
    let id = jobs.submit(spot).expect("syntactically fine").job_id;
    let done = finish(&jobs, id, Duration::from_secs(30));
    assert_eq!(done.phase, Phase::Failed);
    let error = done.error.unwrap();
    assert!(
        error.contains("locks[0]") && error.contains("cannot be applied"),
        "{error}"
    );
}

#[test]
fn a_syntactically_bad_lock_fails_the_command_not_the_job() {
    let mut spot = river("2c7dTh4sQd", "QQ+", 30);
    spot.locks = vec![pkwiz_solver::Lock {
        history: Vec::new(),
        strategy: vec![vec![-0.5, 1.0]],
        hands: None,
    }];
    let jobs = Jobs::new(Arc::new(Silent));
    let err = jobs.submit(spot).unwrap_err();
    assert!(
        err.to_string().contains("locks[0]") && err.to_string().contains("non-negative"),
        "{err}"
    );
    assert!(jobs.list().is_empty(), "no job row was created");
}

#[test]
fn a_hands_guard_catches_an_order_mistake() {
    let mut spot = river("2c7dTh4sQd", "QQ+", 30);
    let (jobs, done) = solve(spot.clone());
    let root = jobs.node(done.job_id, &[]).unwrap();

    let mut wrong_hands = root.hands.clone();
    wrong_hands.reverse();
    let num_actions = root.actions.len();
    spot.locks = vec![pkwiz_solver::Lock {
        history: Vec::new(),
        strategy: vec![vec![1.0; root.hands.len()]; num_actions],
        hands: Some(wrong_hands),
    }];

    let jobs = Jobs::new(Arc::new(Silent));
    let id = jobs.submit(spot).unwrap().job_id;
    let done = finish(&jobs, id, Duration::from_secs(30));
    assert_eq!(done.phase, Phase::Failed);
    assert!(done.error.unwrap().contains("hands["));
}

#[test]
fn ev_detail_rows_match_actions_and_ev_is_their_strategy_weighted_average() {
    let (jobs, done) = solve(river("2c7dTh4sQd", "QQ+,AKs,76s", 200));
    let view = jobs.node(done.job_id, &[]).unwrap();

    assert_eq!(view.ev_detail.len(), view.actions.len());
    for row in &view.ev_detail {
        assert_eq!(row.len(), view.hands.len());
    }

    // The engine computes ev as the strategy-weighted row-average of evDetail; pin the exact
    // identity, which also catches any chunking or row-order mistake.
    for h in 0..view.hands.len() {
        let weighted: f32 = (0..view.actions.len())
            .map(|a| view.strategy[a][h] * view.ev_detail[a][h])
            .sum();
        assert!(
            (weighted - view.ev[h]).abs() < 1e-3,
            "hand {h}: {weighted} vs {}",
            view.ev[h]
        );
    }
}

#[test]
fn a_fold_row_is_exactly_zero_and_empty_nodes_have_no_detail() {
    let (jobs, done) = solve(river("2c7dTh4sQd", "QQ+,AKs,76s", 60));
    let root = jobs.node(done.job_id, &[]).unwrap();
    let bet = root
        .actions
        .iter()
        .position(|a| a.starts_with("Bet"))
        .expect("the root offers a bet");

    let facing = jobs.node(done.job_id, &[bet]).unwrap();
    let fold = facing
        .actions
        .iter()
        .position(|a| a == "Fold")
        .expect("facing a bet offers Fold");
    assert!(
        facing.ev_detail[fold].iter().all(|&v| v == 0.0),
        "fold EV is the engine's exact-zero convention: {:?}",
        facing.ev_detail[fold]
    );

    // Terminal node: no detail, like the other per-hand arrays.
    let terminal = jobs.node(done.job_id, &[bet, fold]).unwrap();
    assert!(terminal.is_terminal);
    assert!(terminal.ev_detail.is_empty());
    assert!(!terminal.is_locked);
}

#[test]
fn an_added_line_shows_up_in_the_node_and_growss_the_estimate() {
    let base = river("2c7dTh4sQd", "QQ+,AKs", 30);
    let mut edited = base.clone();
    edited.added_lines = vec!["Bet(75)".to_owned()];

    let plain = pkwiz_solver::engine::estimate(&base).unwrap();
    let bigger = pkwiz_solver::engine::estimate(&edited).unwrap();
    assert!(
        bigger.allocated > plain.allocated,
        "{} vs {}",
        bigger.allocated,
        plain.allocated
    );

    let (jobs, done) = solve(edited);
    assert_eq!(done.phase, Phase::Done, "{:?}", done.error);
    let root = jobs.node(done.job_id, &[]).unwrap();
    assert!(
        root.actions.iter().any(|a| a == "Bet(75)"),
        "{:?}",
        root.actions
    );
    assert_eq!(root.strategy.len(), root.actions.len());
}

#[test]
fn a_removed_line_disappears_and_shrinks_the_tree() {
    let base = river("2c7dTh4sQd", "QQ+,AKs", 30);
    let (jobs, done) = solve(base.clone());
    let before = jobs.node(done.job_id, &[]).unwrap();
    let bet = before
        .actions
        .iter()
        .find(|a| a.starts_with("Bet"))
        .expect("the default tree offers a bet")
        .clone();

    let mut edited = base.clone();
    edited.removed_lines = vec![bet.clone()];
    assert!(
        pkwiz_solver::engine::estimate(&edited).unwrap().allocated
            < pkwiz_solver::engine::estimate(&base).unwrap().allocated
    );

    let (jobs, done) = solve(edited);
    assert_eq!(done.phase, Phase::Done, "{:?}", done.error);
    let after = jobs.node(done.job_id, &[]).unwrap();
    assert!(!after.actions.contains(&bet), "{:?}", after.actions);
}

#[test]
fn bad_edits_fail_the_command_synchronously_and_name_the_line() {
    let jobs = Jobs::new(Arc::new(Silent));

    let mut spot = river("2c7dTh4sQd", "QQ+", 30);
    spot.removed_lines = vec!["Bet(999)".to_owned()];
    let err = jobs.submit(spot).unwrap_err().to_string();
    assert!(
        err.contains("cannot remove the line `Bet(999)`") && err.contains("does not exist"),
        "{err}"
    );

    for (line, needle) in [
        ("Bet(banana)", "not an amount"),
        ("Chance(Qc)", "chance actions are omitted"),
        ("", "at least one action"),
    ] {
        let mut spot = river("2c7dTh4sQd", "QQ+", 30);
        spot.added_lines = vec![line.to_owned()];
        let err = jobs.submit(spot).unwrap_err().to_string();
        assert!(err.contains(needle), "`{line}`: {err}");
    }

    assert!(jobs.list().is_empty(), "no job rows were created");
}

#[test]
fn removing_every_action_at_a_node_is_refused_with_its_path() {
    let mut spot = river("2c7dTh4sQd", "QQ+", 30);
    let (jobs, done) = solve(spot.clone());
    let root = jobs.node(done.job_id, &[]).unwrap();

    // Remove everything the root offers; the engine would silently accept it and only the
    // game constructor would refuse, namelessly. The sidecar names the node instead.
    spot.removed_lines = root.actions.clone();
    let jobs = Jobs::new(Arc::new(Silent));
    let err = jobs.submit(spot).unwrap_err().to_string();
    assert!(
        err.contains("no actions") && err.contains("(root)"),
        "{err}"
    );
}

#[test]
fn adds_apply_before_removes_so_a_default_line_under_an_added_bet_can_be_pruned() {
    // First learn what the engine builds under an added bet.
    let mut probe = river("2c7dTh4sQd", "QQ+,AKs", 30);
    probe.added_lines = vec!["Bet(75)".to_owned()];
    let (jobs, done) = solve(probe.clone());
    let root = jobs.node(done.job_id, &[]).unwrap();
    let bet_index = root.actions.iter().position(|a| a == "Bet(75)").unwrap();
    let under = jobs.node(done.job_id, &[bet_index]).unwrap();
    let raise = under
        .actions
        .iter()
        .find(|a| a.starts_with("Raise") || a.starts_with("AllIn"))
        .expect("the added bet's subtree got a raise-family action")
        .clone();

    // Now prune that default raise inside the same request that adds the bet.
    let mut edited = probe;
    edited.removed_lines = vec![format!("Bet(75), {raise}")];
    let (jobs, done) = solve(edited);
    assert_eq!(done.phase, Phase::Done, "{:?}", done.error);
    let root = jobs.node(done.job_id, &[]).unwrap();
    let bet_index = root.actions.iter().position(|a| a == "Bet(75)").unwrap();
    let under = jobs.node(done.job_id, &[bet_index]).unwrap();
    assert!(!under.actions.contains(&raise), "{:?}", under.actions);
}

#[test]
fn edited_trees_round_trip_through_a_file() {
    let dir = temp_dir("edited-roundtrip");
    let path = dir.join("edited.bin");

    let mut spot = river("2c7dTh4sQd", "QQ+,AKs", 30);
    spot.added_lines = vec!["Bet(75)".to_owned()];
    spot.save_path = Some(path.to_string_lossy().into_owned());
    let (jobs, done) = solve(spot);
    assert!(done.saved_to.is_some());

    let reopened = jobs.open(&path.to_string_lossy(), None, None).unwrap();
    let root = jobs.node(reopened.job_id, &[]).unwrap();
    assert!(
        root.actions.iter().any(|a| a == "Bet(75)"),
        "the file carried the edit: {:?}",
        root.actions
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn progress_frames_carry_the_convergence_curve_live() {
    let collector = Arc::new(Collector::default());
    let jobs = Jobs::with_throttle(Arc::clone(&collector) as Arc<dyn Emit>, Duration::ZERO);
    let id = jobs
        .submit(river("2c7dTh4sQd", "QQ+,AKs,76s", 200))
        .unwrap()
        .job_id;
    finish(&jobs, id, Duration::from_secs(60));

    // Running frames must carry the curve so far — that is what lets a host draw convergence
    // while the solve runs — and the curve must only ever grow between frames.
    let mut lengths = Vec::new();
    let mut running_with_history = 0;
    for event in collector.events() {
        let job = &event["job"];
        if job["jobId"].as_u64() != Some(id) {
            continue;
        }
        let len = job["history"].as_array().map_or(0, Vec::len);
        lengths.push(len);
        if job["phase"] == "running" && len > 1 {
            running_with_history += 1;
        }
    }
    assert!(
        running_with_history > 0,
        "no running frame carried a curve: {lengths:?}"
    );
    assert!(
        lengths.windows(2).all(|w| w[1] >= w[0] || w[1] == 0),
        "the curve shrank mid-run: {lengths:?}"
    );
}
