//! The `resolve` command: cold re-solve of a job's spot with modified locks/stop.
//!
//! The contracts worth pinning here:
//!
//! 1. **The source is untouched.** A resolve is a new job; the source's status, saved file and
//!    readability must not move.
//! 2. **The merge rules are exactly the documented ones.** Absent `locks` inherits, `[]`
//!    clears, `savePath` is never inherited, `stop` replaces whole.
//! 3. **The seeded bunching Arc outlives the preparation job.** Forgetting the prep must not
//!    break a resolve — the Arc is the data.
//! 4. **Refusals are stable.** Everything without a rebuildable spot answers
//!    `not_resolvable`; nothing else does.
//!
//! Every spot is a tiny river tree (QQ+-class ranges, ≤ 32 iterations); the ICM test reuses
//! the icm.rs situation so its $ numbers have a known baseline.

use std::sync::Arc;
use std::time::{Duration, Instant};

use pkwiz_solver::spot::{BoardSpec, IcmSpec, RangeSpec, Sizing, StreetSizing};
use pkwiz_solver::{
    BunchingRef, BunchingSpec, JobStatus, Jobs, Lock, Phase, ReportKind, ReportSpec, Silent, Spot,
    Stop,
};

/// A river-only spot on `2c7dTh4sQd`: no chance nodes, seconds to solve.
fn river(oop: &str, ip: &str, iterations: u32) -> Spot {
    Spot {
        oop: RangeSpec::Notation(oop.to_owned()),
        ip: RangeSpec::Notation(ip.to_owned()),
        board: BoardSpec::Text("2c7dTh4sQd".to_owned()),
        pot: 100,
        effective_stack: 100,
        sizing: Sizing {
            flop: StreetSizing::none(),
            turn: StreetSizing::none(),
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
    let dir = std::env::temp_dir().join(format!("pkwiz-resolve-{}-{label}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn solve(jobs: &Jobs, spot: Spot) -> JobStatus {
    let queued = jobs.submit(spot).expect("the spot is valid");
    let done = finish(jobs, queued.job_id, Duration::from_secs(120));
    assert_eq!(done.phase, Phase::Done, "{:?}", done.error);
    done
}

/// A lock pinning every hand at the node reached by `history` to action `action`.
fn pin_all(jobs: &Jobs, id: u64, history: &[usize], action: usize) -> Lock {
    let view = jobs.node(id, history).unwrap();
    let mut strategy = vec![vec![0.0f32; view.hands.len()]; view.actions.len()];
    strategy[action] = vec![1.0; view.hands.len()];
    Lock {
        history: history.to_vec(),
        strategy,
        hands: None,
    }
}

#[test]
fn a_resolve_is_a_new_job_and_the_source_does_not_move() {
    let jobs = Jobs::new(Arc::new(Silent));
    let done = solve(&jobs, river("QQ+,AKs", "QQ+,AKs", 32));
    let source = done.job_id;
    assert_eq!(done.resolved_from, None, "an ordinary solve has no parent");
    assert!(!jobs.node(source, &[]).unwrap().is_locked);

    let lock = pin_all(&jobs, source, &[], 0);
    let queued = jobs
        .resolve(source, Some(vec![lock]), None, None, None, None)
        .unwrap();
    assert_eq!(queued.resolved_from, Some(source));
    assert_eq!(queued.kind, pkwiz_solver::JobKind::Solve);

    let resolved = finish(&jobs, queued.job_id, Duration::from_secs(120));
    assert_eq!(resolved.phase, Phase::Done, "{:?}", resolved.error);
    assert_eq!(resolved.resolved_from, Some(source));

    // The locked node reads back pinned, at exactly the pinned (already-normalized) mix.
    let root = jobs.node(queued.job_id, &[]).unwrap();
    assert!(root.is_locked);
    for h in 0..root.hands.len() {
        assert!(
            (root.strategy[0][h] - 1.0).abs() < 1e-6,
            "hand {h}: {:?}",
            root.strategy
        );
    }

    // The source did not move: status equal field for field, still readable, still unlocked.
    let after = jobs.status(source).unwrap();
    assert_eq!(after, done);
    assert!(!jobs.node(source, &[]).unwrap().is_locked);

    // A resolve of a resolve chains, naming the immediate parent only.
    let chained = jobs
        .resolve(queued.job_id, None, None, None, None, None)
        .unwrap();
    assert_eq!(chained.resolved_from, Some(queued.job_id));
    let chained = finish(&jobs, chained.job_id, Duration::from_secs(120));
    assert_eq!(chained.phase, Phase::Done, "{:?}", chained.error);
    // Absent locks inherited the parent's lock.
    assert!(jobs.node(chained.job_id, &[]).unwrap().is_locked);
}

#[test]
fn absent_locks_inherit_and_empty_locks_clear_over_the_wire() {
    // Driven through the wire deliberately: the absent-vs-`[]` distinction is a serde
    // contract (`Option<Vec<Lock>>` with `#[serde(default)]`), and only a JSON round trip
    // proves it.
    let jobs = Jobs::new(Arc::new(Silent));
    let mut spot = river("QQ+", "QQ+", 16);
    let queued = jobs.submit(spot.clone()).unwrap();
    finish(&jobs, queued.job_id, Duration::from_secs(120));
    spot.locks = vec![pin_all(&jobs, queued.job_id, &[], 0)];
    let source = solve(&jobs, spot).job_id;
    assert!(jobs.node(source, &[]).unwrap().is_locked);

    let resolve = |extra: &str| -> u64 {
        let command: pkwiz_solver::Command =
            serde_json::from_str(&format!(r#"{{"cmd":"resolve","jobId":{source}{extra}}}"#))
                .unwrap();
        let result = pkwiz_solver::execute(&jobs, command).unwrap();
        assert_eq!(result["resolvedFrom"].as_u64(), Some(source));
        result["jobId"].as_u64().unwrap()
    };

    // Absent: the source's lock still applies in the new job.
    let inherited = resolve("");
    assert_eq!(
        finish(&jobs, inherited, Duration::from_secs(120)).phase,
        Phase::Done
    );
    assert!(jobs.node(inherited, &[]).unwrap().is_locked);

    // `[]`: the previously locked node reads back free.
    let cleared = resolve(r#","locks":[]"#);
    assert_eq!(
        finish(&jobs, cleared, Duration::from_secs(120)).phase,
        Phase::Done
    );
    assert!(!jobs.node(cleared, &[]).unwrap().is_locked);
}

#[test]
fn a_stop_override_replaces_whole_with_solve_defaults_for_absent_fields() {
    let jobs = Jobs::new(Arc::new(Silent));
    // The source runs its full 16 iterations (unreachable target); the resolve's partial stop
    // must not merge against that — absent fields take `Stop::default()`, so
    // `checkInterval` becomes 10 and the percentage target 0.5, exactly as a fresh solve.
    let source = solve(&jobs, river("QQ+", "QQ+", 16)).job_id;

    let command: pkwiz_solver::Command = serde_json::from_str(&format!(
        r#"{{"cmd":"resolve","jobId":{source},"stop":{{"maxIterations":500}}}}"#
    ))
    .unwrap();
    let result = pkwiz_solver::execute(&jobs, command).unwrap();
    let resolved = result["jobId"].as_u64().unwrap();
    assert_eq!(result["maxIterations"].as_u64(), Some(500));
    // The default 0.5% of a 100 pot, not the source's absolute-zero target.
    assert!((result["targetExploitability"].as_f64().unwrap() - 0.5).abs() < 1e-9);

    let done = finish(&jobs, resolved, Duration::from_secs(120));
    assert_eq!(done.phase, Phase::Done, "{:?}", done.error);
}

#[test]
fn save_path_is_never_inherited_and_the_sources_file_survives() {
    let dir = temp_dir("save-path");
    let path = dir.join("source.bin");

    let jobs = Jobs::new(Arc::new(Silent));
    let mut spot = river("QQ+,AKs", "QQ+,AKs", 16);
    spot.save_path = Some(path.to_string_lossy().into_owned());
    let done = solve(&jobs, spot);
    let source = done.job_id;
    assert!(done.saved_to.is_some());
    let before = std::fs::read(&path).unwrap();

    let queued = jobs.resolve(source, None, None, None, None, None).unwrap();
    let resolved = finish(&jobs, queued.job_id, Duration::from_secs(120));
    assert_eq!(resolved.phase, Phase::Done, "{:?}", resolved.error);
    assert_eq!(
        resolved.saved_to, None,
        "an uninherited savePath means in-memory only"
    );

    // The source's file was not rewritten — byte for byte.
    assert_eq!(std::fs::read(&path).unwrap(), before);
    // And the source still reloads from it (its tree was swept when the resolve started).
    assert!(jobs.node(source, &[]).is_ok());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn everything_without_a_rebuildable_spot_answers_not_resolvable() {
    let dir = temp_dir("refusals");
    let jobs = Jobs::new(Arc::new(Silent));
    let path = dir.join("source.bin");
    let mut spot = river("QQ+", "QQ+", 16);
    spot.save_path = Some(path.to_string_lossy().into_owned());
    let source = solve(&jobs, spot).job_id;

    // Unknown id: the row lookup fails first.
    assert_eq!(
        jobs.resolve(9999, None, None, None, None, None)
            .unwrap_err()
            .code(),
        "no_such_job"
    );

    // A bunching preparation, refused by kind before it even runs.
    let prep: BunchingSpec =
        serde_json::from_str(r#"{"foldRanges":["AA"],"flop":"2c7dTh"}"#).unwrap();
    let prep_id = jobs.submit_bunching(prep).unwrap().job_id;
    let err = jobs
        .resolve(prep_id, None, None, None, None, None)
        .unwrap_err();
    assert_eq!(err.code(), "not_resolvable");
    assert!(err.to_string().contains("bunching"), "{err}");
    jobs.cancel(prep_id).unwrap();

    // Report, dump and simplify jobs: analyses have no spot.
    let report_id = jobs
        .submit_report(ReportSpec {
            job_id: source,
            history: Vec::new(),
            kind: ReportKind::Lines,
            line: Vec::new(),
            categories: false,
        })
        .unwrap()
        .job_id;
    let dump_id = jobs
        .submit_dump(pkwiz_solver::DumpSpec {
            job_id: source,
            path: dir.join("d.jsonl").to_string_lossy().into_owned(),
            history: Vec::new(),
            max_board_cards: 5,
            include: pkwiz_solver::DumpInclude::default(),
            max_bytes: None,
            compress: false,
            compression_level: Spot::default_compression_level(),
        })
        .unwrap()
        .job_id;
    let simplify_id = jobs
        .submit_simplify(pkwiz_solver::SimplifySpec {
            job_id: source,
            player: 0,
            history: Vec::new(),
            max_board_cards: 3,
            grid: 2,
            purify_threshold: None,
            include_hands: false,
        })
        .unwrap()
        .job_id;
    for (id, kind) in [
        (report_id, "report"),
        (dump_id, "dump"),
        (simplify_id, "simplify"),
    ] {
        let err = jobs.resolve(id, None, None, None, None, None).unwrap_err();
        assert_eq!(err.code(), "not_resolvable", "{kind}");
        assert!(err.to_string().contains(kind), "{err}");
    }

    // An opened job: its placeholder spot cannot rebuild a game.
    let opened = jobs
        .open(&path.to_string_lossy(), None, None, None)
        .unwrap();
    let err = jobs
        .resolve(opened.job_id, None, None, None, None, None)
        .unwrap_err();
    assert_eq!(err.code(), "not_resolvable");
    assert!(err.to_string().contains("open"), "{err}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_source_cancelled_while_queued_still_resolves() {
    // The gate is on the spot, not the phase: a job cancelled before it ever ran has a
    // complete spot from submit time.
    let jobs = Jobs::new(Arc::new(Silent));
    let blocker = jobs
        .submit(river("QQ+,AKs,76s", "QQ+,AKs,76s", 1_000_000))
        .unwrap()
        .job_id;
    let victim = jobs.submit(river("QQ+", "QQ+", 16)).unwrap().job_id;
    let cancelled = jobs.cancel(victim).unwrap();
    assert_eq!(cancelled.phase, Phase::Cancelled, "cancelled while queued");
    jobs.cancel(blocker).unwrap();
    finish(&jobs, blocker, Duration::from_secs(60));

    let queued = jobs.resolve(victim, None, None, None, None, None).unwrap();
    let resolved = finish(&jobs, queued.job_id, Duration::from_secs(120));
    assert_eq!(resolved.phase, Phase::Done, "{:?}", resolved.error);
    assert_eq!(resolved.resolved_from, Some(victim));
}

#[test]
fn a_failed_source_resolves_with_corrected_locks() {
    let jobs = Jobs::new(Arc::new(Silent));
    // Syntactically fine, structurally wrong: the root offers more than one action and holds
    // more than one hand, so this 1×1 lock fails the *job*, exactly like `solve`.
    let mut spot = river("QQ+", "QQ+", 16);
    spot.locks = vec![Lock {
        history: Vec::new(),
        strategy: vec![vec![1.0]],
        hands: None,
    }];
    let queued = jobs.submit(spot).unwrap();
    let failed = finish(&jobs, queued.job_id, Duration::from_secs(120));
    assert_eq!(failed.phase, Phase::Failed);
    assert!(failed.error.unwrap().contains("locks[0]"));

    // Corrected (here: cleared) locks re-solve the same spot to completion.
    let queued = jobs
        .resolve(failed.job_id, Some(Vec::new()), None, None, None, None)
        .unwrap();
    let resolved = finish(&jobs, queued.job_id, Duration::from_secs(120));
    assert_eq!(resolved.phase, Phase::Done, "{:?}", resolved.error);
    assert!(!jobs.node(queued.job_id, &[]).unwrap().is_locked);
}

#[test]
fn a_forgotten_preparation_does_not_break_a_resolve_the_seeded_arc_is_the_data() {
    let jobs = Jobs::new(Arc::new(Silent));
    let prep: BunchingSpec =
        serde_json::from_str(r#"{"foldRanges":["AA"],"flop":"2c7dTh"}"#).unwrap();
    let prep_id = jobs.submit_bunching(prep).unwrap().job_id;
    assert_eq!(
        finish(&jobs, prep_id, Duration::from_secs(120)).phase,
        Phase::Done
    );

    let mut spot = river("QQ+", "QQ+", 16);
    spot.bunching = Some(BunchingRef::Job { job_id: prep_id });
    let source = solve(&jobs, spot).job_id;
    let source_root = jobs.node(source, &[]).unwrap();

    // The preparation was never saved; forgetting it leaves the solve's retained Arc as the
    // only copy of the data.
    jobs.forget(prep_id).unwrap();

    let queued = jobs.resolve(source, None, None, None, None, None).unwrap();
    let resolved = finish(&jobs, queued.job_id, Duration::from_secs(120));
    assert_eq!(resolved.phase, Phase::Done, "{:?}", resolved.error);

    // The root's weights and equities are strategy-independent and bunching-sensitive, so
    // equality here proves the resolve was built on the same data the source used — not on a
    // silently non-bunching tree.
    let resolved_root = jobs.node(queued.job_id, &[]).unwrap();
    assert_eq!(resolved_root.weights, source_root.weights);
    assert_eq!(resolved_root.equity, source_root.equity);
}

#[test]
fn an_icm_source_resolves_in_dollars_with_the_same_pot_value() {
    // The icm.rs situation: two contestants and a big-stacked bystander, KK vs AA-heavy, an
    // all-in line, so $ EVs are non-trivial and the OOP baseline sits near $262.
    let jobs = Jobs::new(Arc::new(Silent));
    let mut spot = river("KK", "AA,JJ", 200);
    spot.sizing.river = StreetSizing {
        bet: "50%".to_owned(),
        raise: "2.5x".to_owned(),
    };
    spot.stop = Stop {
        max_iterations: 200,
        target_exploitability: None,
        target_exploitability_pct: 0.5,
        check_interval: 5,
    };
    spot.icm = Some(IcmSpec {
        payouts: vec![500.0, 300.0, 200.0],
        stacks: vec![100.0, 400.0, 500.0],
        oop_seat: 0,
        ip_seat: 1,
    });
    let done = solve(&jobs, spot);
    let source = done.job_id;
    let pot_value = done.icm_pot_value.expect("an ICM job carries icmPotValue");

    // Resolve with a lock — ICM is tree-adjacent configuration and rides along unchanged.
    let lock = pin_all(&jobs, source, &[], 0);
    let queued = jobs
        .resolve(source, Some(vec![lock]), None, None, None, None)
        .unwrap();
    assert_eq!(queued.icm_pot_value, Some(pot_value));
    let resolved = finish(&jobs, queued.job_id, Duration::from_secs(120));
    assert_eq!(resolved.phase, Phase::Done, "{:?}", resolved.error);
    assert_eq!(resolved.icm_pot_value, Some(pot_value));

    // The resolved EVs are $, not chips: OOP's whole-range average sits near its ~$262
    // tournament baseline, nowhere near a 100-chip pot.
    let root = jobs.node(queued.job_id, &[]).unwrap();
    assert!(root.is_locked);
    let average_ev = root.average_ev.unwrap();
    assert!(
        (200.0..320.0).contains(&average_ev),
        "expected an absolute $ EV near the baseline, got {average_ev}"
    );
    let ev = resolved.ev.unwrap();
    assert!(ev[0] != 0.0 || ev[1] != 0.0, "a $ delta exists: {ev:?}");
}

#[test]
fn a_job_status_without_resolved_from_still_deserializes() {
    // The PROTOCOL_VERSION 1 guard, `resolvedFrom` edition: frames stored before the field
    // existed must still parse, and jobs not created by `resolve` must not carry it at all.
    let jobs = Jobs::new(Arc::new(Silent));
    let done = solve(&jobs, river("QQ+", "QQ+", 8));
    let json = serde_json::to_value(&done).unwrap();
    assert!(
        !json.as_object().unwrap().contains_key("resolvedFrom"),
        "skip_serializing_if keeps old frames byte-identical"
    );
    let parsed: JobStatus = serde_json::from_value(json).unwrap();
    assert_eq!(parsed.resolved_from, None);
    assert_eq!(parsed, done);

    // A resolved job's status carries it and round-trips.
    let queued = jobs
        .resolve(done.job_id, None, None, None, None, None)
        .unwrap();
    let resolved = finish(&jobs, queued.job_id, Duration::from_secs(120));
    let json = serde_json::to_value(&resolved).unwrap();
    assert_eq!(json["resolvedFrom"].as_u64(), Some(done.job_id));
    let back: JobStatus = serde_json::from_value(json).unwrap();
    assert_eq!(back, resolved);
}
