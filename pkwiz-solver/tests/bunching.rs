//! The bunching effect, over the wire and around the job lifecycle.
//!
//! What is worth asserting is not the engine's arithmetic — its own suite pins that against
//! known numbers — but the sidecar's contracts around it:
//!
//! 1. **A preparation is a job.** It streams monotonic progress, cancels promptly, saves and
//!    reopens, and a cancelled one keeps *nothing* (a half-computed table is unusable, unlike a
//!    half-converged strategy).
//! 2. **The data actually reaches the solve.** The same spot solved with and without a
//!    preparation must answer differently — otherwise every other test here is theater.
//! 3. **Reload paths re-apply the effect.** The engine's file format records no bunching, so
//!    the one silent-corruption path in this feature is a released job reloading its game and
//!    forgetting `set_bunching_effect`. `release_then_node_reapplies_the_effect` is the
//!    regression test standing in front of that.
//!
//! Every preparation here uses one fold player — the same configuration the engine's own fast
//! test (`test_bunching_independent_1`) uses, seconds instead of minutes — except the cancel
//! test, which needs two so there is a mid-run to cancel in.

use std::sync::Arc;
use std::time::{Duration, Instant};

use pkwiz_solver::spot::{BoardSpec, RangeSpec, Sizing, StreetSizing};
use pkwiz_solver::{
    BunchingRef, BunchingSpec, Collector, Emit, JobKind, JobStatus, Jobs, Phase, Silent, Spot,
    Stop, Stopped,
};

fn prep(fold_ranges: &[&str], flop: &str) -> BunchingSpec {
    BunchingSpec {
        fold_ranges: fold_ranges
            .iter()
            .map(|r| RangeSpec::Notation((*r).to_owned()))
            .collect(),
        flop: BoardSpec::Text(flop.to_owned()),
        max_memory_bytes: None,
        save_path: None,
        compression_level: Some(3),
        memo: None,
    }
}

/// A river spot whose flop is `2c7dTh`, so it can reference a preparation on that flop. The
/// ranges are chosen to *interact* with a fold range of `AA`: hero's KK loses to exactly the
/// AA half of villain's range, so folding players who held aces visibly shifts the answer.
fn river_spot(iterations: u32) -> Spot {
    Spot {
        oop: RangeSpec::Notation("KK".to_owned()),
        ip: RangeSpec::Notation("AA,JJ".to_owned()),
        board: BoardSpec::Text("2c7dTh4sQd".to_owned()),
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
        icm: None,
    }
}

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

fn temp_dir(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("pkwiz-bunching-{}-{label}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn a_one_player_preparation_completes_with_monotonic_progress() {
    let collector = Arc::new(Collector::default());
    let jobs = Jobs::with_throttle(Arc::clone(&collector) as Arc<dyn Emit>, Duration::ZERO);

    let queued = jobs.submit_bunching(prep(&["AA"], "2s2h2d")).unwrap();
    assert_eq!(queued.kind, JobKind::Bunching);
    assert_eq!(queued.phase, Phase::Queued);
    let initial = queued.bunching.as_ref().expect("bunching jobs carry it");
    assert_eq!((initial.stage, initial.overall_percent), (1, 0));
    assert_eq!(initial.fold_players, 1);
    assert_eq!(initial.flop, ["2d", "2h", "2s"]);

    let done = finish(&jobs, queued.job_id, Duration::from_secs(120));
    assert_eq!(done.phase, Phase::Done, "{:?}", done.error);
    assert!(done.resident);
    assert_eq!(done.stopped, None, "why-a-loop-ended is a solve concept");
    let bunching = done.bunching.expect("present at done");
    assert_eq!((bunching.stage, bunching.overall_percent), (3, 100));
    // The three result tables alone are 64 410 304 bytes; anything below that means the data
    // was not actually computed.
    assert!(bunching.memory_bytes.unwrap() > 60_000_000);

    let events = collector.events();
    assert_eq!(events[0]["job"]["phase"], "queued");
    assert_eq!(events.last().unwrap()["job"]["phase"], "done");
    let phases: Vec<&str> = events
        .iter()
        .map(|e| e["job"]["phase"].as_str().unwrap())
        .collect();
    assert!(
        phases.contains(&"building") && phases.contains(&"running"),
        "{phases:?}"
    );

    // The progress bar's contract: overall percent never goes backwards, and the stages arrive
    // in order 1 → 2 → 3 across the run.
    let progress: Vec<(u64, u64)> = events
        .iter()
        .filter(|e| !e["job"]["bunching"].is_null())
        .map(|e| {
            (
                e["job"]["bunching"]["stage"].as_u64().unwrap(),
                e["job"]["bunching"]["overallPercent"].as_u64().unwrap(),
            )
        })
        .collect();
    assert!(
        progress.windows(2).all(|w| w[0] <= w[1]),
        "progress went backwards: {progress:?}"
    );
    for stage in [1, 2, 3] {
        assert!(
            progress.iter().any(|(s, _)| *s == stage),
            "no frame from stage {stage}: {progress:?}"
        );
    }
}

#[test]
fn a_bunching_solve_changes_the_answer() {
    let jobs = Jobs::new(Arc::new(Silent));

    let plain = finish(
        &jobs,
        jobs.submit(river_spot(60)).unwrap().job_id,
        Duration::from_secs(60),
    );
    assert_eq!(plain.phase, Phase::Done, "{:?}", plain.error);
    let plain_root = jobs.node(plain.job_id, &[]).unwrap();

    let prepared = finish(
        &jobs,
        jobs.submit_bunching(prep(&["AA"], "2c7dTh"))
            .unwrap()
            .job_id,
        Duration::from_secs(120),
    );
    assert_eq!(prepared.phase, Phase::Done, "{:?}", prepared.error);

    let mut spot = river_spot(60);
    spot.bunching = Some(BunchingRef::Job {
        job_id: prepared.job_id,
    });
    let bunched = finish(
        &jobs,
        jobs.submit(spot).unwrap().job_id,
        Duration::from_secs(60),
    );
    assert_eq!(bunched.phase, Phase::Done, "{:?}", bunched.error);
    let bunched_root = jobs.node(bunched.job_id, &[]).unwrap();

    // A player who folded held aces, so villain's AA half shrinks and hero's KK gains equity —
    // if these agree, the data never reached the solve.
    let plain_equity = plain_root.average_equity.unwrap();
    let bunched_equity = bunched_root.average_equity.unwrap();
    assert!(
        (plain_equity - bunched_equity).abs() > 1e-3,
        "bunching changed nothing: {plain_equity} vs {bunched_equity}"
    );
    assert!(
        (plain_root.average_ev.unwrap() - bunched_root.average_ev.unwrap()).abs() > 1e-3,
        "EV did not move either"
    );
}

#[test]
fn cancel_mid_preparation_leaves_no_data() {
    let jobs = Jobs::new(Arc::new(Silent));
    // Two fold players: enough work that `running` reliably has a mid-run to cancel in.
    let queued = jobs
        .submit_bunching(prep(
            &[
                "22+,A2s+,A2o+,K2s+,K2o+,Q2s+,Q2o+",
                "22+,A2s+,A2o+,K2s+,K2o+,Q2s+,Q2o+",
            ],
            "2s2h2d",
        ))
        .unwrap();
    let id = queued.job_id;

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let status = jobs.status(id).unwrap();
        if status.phase == Phase::Running {
            break;
        }
        assert!(Instant::now() < deadline, "never started: {status:?}");
        std::thread::sleep(Duration::from_millis(1));
    }
    jobs.cancel(id).unwrap();

    let done = finish(&jobs, id, Duration::from_secs(30));
    assert_eq!(done.phase, Phase::Cancelled);
    assert_eq!(done.stopped, Some(Stopped::Cancelled));
    assert!(!done.resident, "a cancelled preparation keeps nothing");

    // Nothing to save — unlike a cancelled solve, whose finalized strategy is still worth a
    // file — and a solve referencing it is refused at submit, because the prep is already
    // terminal with no data.
    let refused = jobs.save(id, "/tmp/never-written.bunching").unwrap_err();
    assert_eq!(refused.code(), "nothing_to_save");

    let mut spot = river_spot(10);
    spot.board = BoardSpec::Text("2s2h2d7cQd".to_owned());
    spot.bunching = Some(BunchingRef::Job { job_id: id });
    let err = jobs.submit(spot).unwrap_err();
    assert_eq!(err.code(), "bunching_not_ready", "{err}");
}

#[test]
fn a_flop_mismatch_is_synchronous_for_job_refs() {
    let jobs = Jobs::new(Arc::new(Silent));
    let prep_id = jobs
        .submit_bunching(prep(&["AA"], "Td9d6h"))
        .unwrap()
        .job_id;

    let mut spot = river_spot(10);
    spot.board = BoardSpec::Text("QsJh2h".to_owned());
    spot.bunching = Some(BunchingRef::Job { job_id: prep_id });
    let err = jobs.submit(spot).unwrap_err();
    assert_eq!(err.code(), "engine");
    let message = err.to_string();
    // Both flops, sorted the way the engine sorts them, so the message is actionable.
    assert!(
        message.contains("6h9dTd") && message.contains("2hJhQs"),
        "{message}"
    );
    // No solve job row was created; only the preparation exists.
    assert_eq!(jobs.list().len(), 1);

    jobs.cancel(prep_id).unwrap();
    finish(&jobs, prep_id, Duration::from_secs(30));
}

#[test]
fn an_asymmetric_fold_range_is_refused_synchronously() {
    let jobs = Jobs::new(Arc::new(Silent));
    // A suit-specific combo cannot be suit-symmetric; the engine is the authority and its
    // message travels verbatim under the `engine` code.
    let err = jobs.submit_bunching(prep(&["AhKd"], "Td9d6h")).unwrap_err();
    assert_eq!(err.code(), "engine");
    assert!(err.to_string().contains("suit-symmetric"), "{err}");
    assert!(jobs.list().is_empty(), "no job row was created");
}

#[test]
fn the_peak_memory_gate_refuses_before_allocating() {
    let jobs = Jobs::new(Arc::new(Silent));
    let mut spec = prep(&["AA", "KK", "QQ", "JJ"], "Td9d6h");
    spec.max_memory_bytes = Some(1_000_000);

    // Four fold players peak at ~3.7 GB of temporary tables; the refusal has to come from
    // arithmetic, not from trying.
    let started = Instant::now();
    let err = jobs.submit_bunching(spec).unwrap_err();
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "the gate computed instead of refusing"
    );
    assert_eq!(err.code(), "engine");
    let message = err.to_string();
    assert!(
        message.contains("1000000") && message.contains("4 fold players"),
        "{message}"
    );
    assert!(jobs.list().is_empty());
}

#[test]
fn a_saved_preparation_reopens_and_serves_a_solve() {
    let dir = temp_dir("reopen");
    let path = dir.join("prep.bunching");
    let path_text = path.to_string_lossy().into_owned();

    let jobs = Jobs::new(Arc::new(Silent));
    let mut spec = prep(&["AA"], "2c7dTh");
    spec.save_path = Some(path_text.clone());
    spec.memo = Some("UTG folds".to_owned());
    let saved = finish(
        &jobs,
        jobs.submit_bunching(spec).unwrap().job_id,
        Duration::from_secs(120),
    );
    assert_eq!(saved.phase, Phase::Done, "{:?}", saved.error);
    assert_eq!(saved.saved_to.as_deref(), Some(path_text.as_str()));
    assert!(saved.error.is_none(), "{:?}", saved.error);

    // The file round-trips through the engine's loader as ready data, memo intact.
    let (data, memo) = pkwiz_solver::engine::load_bunching(&path_text, None).unwrap();
    assert!(data.is_ready());
    assert_eq!(memo, "UTG folds");
    drop(data);

    // Forget the preparation entirely; the file is now the only source.
    jobs.forget(saved.job_id).unwrap();

    let reopened = jobs.open_bunching(&path_text, None).unwrap();
    assert_eq!(reopened.phase, Phase::Done);
    assert_eq!(reopened.kind, JobKind::Bunching);
    assert!(reopened.resident);
    assert_eq!(reopened.saved_to.as_deref(), Some(path_text.as_str()));
    assert!(reopened.bunching.unwrap().memory_bytes.unwrap() > 60_000_000);

    let mut by_job = river_spot(40);
    by_job.bunching = Some(BunchingRef::Job {
        job_id: reopened.job_id,
    });
    let solved_by_job = finish(
        &jobs,
        jobs.submit(by_job).unwrap().job_id,
        Duration::from_secs(60),
    );
    assert_eq!(
        solved_by_job.phase,
        Phase::Done,
        "{:?}",
        solved_by_job.error
    );

    let mut by_file = river_spot(40);
    by_file.bunching = Some(BunchingRef::File {
        path: path_text.clone(),
    });
    let solved_by_file = finish(
        &jobs,
        jobs.submit(by_file).unwrap().job_id,
        Duration::from_secs(60),
    );
    assert_eq!(
        solved_by_file.phase,
        Phase::Done,
        "{:?}",
        solved_by_file.error
    );

    // Same data, same spot: the two reference forms must agree. Tolerance rather than
    // bit-equality, because these are two independent CFR runs.
    let a = jobs.node(solved_by_job.job_id, &[]).unwrap();
    let b = jobs.node(solved_by_file.job_id, &[]).unwrap();
    assert!((a.average_ev.unwrap() - b.average_ev.unwrap()).abs() < 1e-3);
    assert!((a.average_equity.unwrap() - b.average_equity.unwrap()).abs() < 1e-5);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn release_then_node_reapplies_the_effect() {
    // THE regression test for the engine's format carrying no bunching field: a released
    // bunching solve reloads its game from disk, and if the reload path forgets
    // `set_bunching_effect`, the numbers silently revert to non-bunching. Bit-identical
    // answers before and after are the only acceptable outcome.
    let dir = temp_dir("release-reapply");
    let path = dir.join("bunched-solve.bin");

    let jobs = Jobs::new(Arc::new(Silent));
    let prepared = finish(
        &jobs,
        jobs.submit_bunching(prep(&["AA"], "2c7dTh"))
            .unwrap()
            .job_id,
        Duration::from_secs(120),
    );
    assert_eq!(prepared.phase, Phase::Done, "{:?}", prepared.error);

    let mut spot = river_spot(40);
    spot.bunching = Some(BunchingRef::Job {
        job_id: prepared.job_id,
    });
    spot.save_path = Some(path.to_string_lossy().into_owned());
    let solved = finish(
        &jobs,
        jobs.submit(spot).unwrap().job_id,
        Duration::from_secs(60),
    );
    assert_eq!(solved.phase, Phase::Done, "{:?}", solved.error);

    let before = jobs.node(solved.job_id, &[]).unwrap();

    // Forgetting the preparation first proves the solve job's retained Arc — not the prep row —
    // is what survives release.
    jobs.forget(prepared.job_id).unwrap();
    let released = jobs.release(solved.job_id).unwrap();
    assert!(!released.resident);

    let after = jobs.node(solved.job_id, &[]).unwrap();
    assert_eq!(after.strategy, before.strategy);
    assert_eq!(after.weights, before.weights);
    assert_eq!(after.equity, before.equity);
    assert_eq!(after.ev, before.ev);
    assert_eq!(after.average_equity, before.average_equity);
    assert_eq!(after.average_ev, before.average_ev);
    assert!(
        jobs.status(solved.job_id).unwrap().resident,
        "reading a released job puts it back in memory"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn node_on_a_bunching_job_is_not_readable() {
    let jobs = Jobs::new(Arc::new(Silent));
    let id = jobs
        .submit_bunching(prep(&["AA"], "Td9d6h"))
        .unwrap()
        .job_id;

    let err = jobs.node(id, &[]).unwrap_err();
    assert_eq!(err.code(), "not_readable");
    assert!(err.to_string().contains("bunching"), "{err}");

    jobs.cancel(id).unwrap();
    finish(&jobs, id, Duration::from_secs(30));
}

#[test]
fn estimate_reports_the_bunching_arena_without_resolving_the_ref() {
    // The extra bytes depend only on the tree, so `estimate` answers even for a reference that
    // points nowhere — a UI can show the cost while the preparation is still queued.
    let base = river_spot(10);
    let mut with = base.clone();
    with.bunching = Some(BunchingRef::Job { job_id: 424242 });

    let plain = pkwiz_solver::engine::estimate(&base).unwrap();
    assert_eq!(plain.bunching_extra, None);

    let bunched = pkwiz_solver::engine::estimate(&with).unwrap();
    let extra = bunched.bunching_extra.expect("present when the spot asks");
    assert!(extra > 0);
    assert_eq!(bunched.allocated, plain.allocated + extra);
}

#[test]
fn opening_the_wrong_file_kind_is_an_error_not_a_crash() {
    let dir = temp_dir("wrong-kind");
    let game_path = dir.join("game.bin");
    let bunching_path = dir.join("data.bunching");

    let collector = Arc::new(Collector::default());
    let session = pkwiz_solver::Session::new(Arc::clone(&collector) as Arc<dyn Emit>);
    let call = |line: &str| -> serde_json::Value {
        let handled = session.handle_line(line);
        serde_json::from_str(&pkwiz_solver::encode(&handled.response)).unwrap()
    };

    // A real game file, via the wire so the whole path is exercised.
    let spot = serde_json::json!({
        "oop": "QQ+", "ip": "QQ+", "board": "2c7dTh4sQd", "pot": 100, "effectiveStack": 100,
        "stop": {"maxIterations": 10, "checkInterval": 5},
        "savePath": game_path.to_string_lossy(),
    });
    let id = call(&format!(r#"{{"cmd":"solve","spot":{spot}}}"#))["result"]["jobId"]
        .as_u64()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let v = call(&format!(r#"{{"cmd":"progress","jobId":{id}}}"#));
        if v["result"]["phase"] == "done" {
            break;
        }
        assert!(Instant::now() < deadline, "solve never finished: {v}");
        std::thread::sleep(Duration::from_millis(5));
    }

    // A real bunching file, prepared directly — the job machinery is not what this test is
    // about.
    let mut data = prep(&["AA"], "2c7dTh").validate().unwrap();
    data.phase1(false);
    data.phase2(false);
    data.phase3(false);
    pkwiz_solver::engine::save_bunching(&data, &bunching_path.to_string_lossy(), "", Some(3))
        .unwrap();

    // Crossed over, both ways: the engine's data-type byte tells them apart and the session
    // reports it instead of dying.
    let v = call(&format!(
        r#"{{"cmd":"openBunching","path":{}}}"#,
        serde_json::json!(game_path.to_string_lossy())
    ));
    assert_eq!(v["ok"], false);
    assert_eq!(v["error"]["code"], "engine");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Data type"),
        "{v}"
    );

    let v = call(&format!(
        r#"{{"cmd":"open","path":{}}}"#,
        serde_json::json!(bunching_path.to_string_lossy())
    ));
    assert_eq!(v["ok"], false);
    assert_eq!(v["error"]["code"], "engine");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Data type"),
        "{v}"
    );

    assert_eq!(call(r#"{"cmd":"ping"}"#)["result"]["pong"], true);

    std::fs::remove_dir_all(&dir).ok();
}
