//! The `simplify` job, and the resolve composition that gives its output meaning.
//!
//! The claims worth pinning:
//!
//! 1. **The headline round trip.** solve → simplify → every emitted column is on-grid and
//!    sums to one → feed the locks to `resolve` → the resolved job is `done`, its locked
//!    nodes read back at exactly the lock values, and its exploited EV cannot beat the
//!    equilibrium it simplified — that resolved EV *is* the simplification's cost.
//! 2. **The rounding rules are the documented ones.** Purify judges original probabilities;
//!    grid 1 purifies everything; zero-weight hands stay free (all-zero columns).
//! 3. **The bounds hold.** Only the requested player's nodes are locked; the street bound
//!    truncates; the lifecycle (cancel, `reportResult`, refusals) mirrors reports.
//!
//! Numeric edge cases of the rounding itself live as unit tests next to `round_column`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use pkwiz_solver::spot::{BoardSpec, RangeSpec, Sizing, StreetSizing};
use pkwiz_solver::{
    Collector, Emit, JobStatus, Jobs, Lock, Phase, Silent, SimplifySpec, Spot, Stop, Stopped,
};

/// A spot on any street, 50%/2.5x on every present street.
fn spot(board: &str, oop: &str, ip: &str, iterations: u32) -> Spot {
    Spot {
        oop: RangeSpec::Notation(oop.to_owned()),
        ip: RangeSpec::Notation(ip.to_owned()),
        board: BoardSpec::Text(board.to_owned()),
        pot: 100,
        effective_stack: 100,
        sizing: Sizing::default(),
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

/// A river-only spot: no chance nodes at all.
fn river(oop: &str, ip: &str, iterations: u32) -> Spot {
    let mut spot = spot("2c7dTh4sQd", oop, ip, iterations);
    spot.sizing.flop = StreetSizing::none();
    spot.sizing.turn = StreetSizing::none();
    spot
}

fn simplify_spec(job_id: u64, player: usize) -> SimplifySpec {
    SimplifySpec {
        job_id,
        player,
        history: Vec::new(),
        max_board_cards: 3,
        grid: 2,
        purify_threshold: Some(0.8),
        include_hands: false,
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

fn solve(jobs: &Jobs, spot: Spot) -> JobStatus {
    let queued = jobs.submit(spot).expect("the spot is valid");
    let done = finish(jobs, queued.job_id, Duration::from_secs(300));
    assert_eq!(done.phase, Phase::Done, "{:?}", done.error);
    done
}

/// Run one simplify job to `done` and hand back its result.
fn simplify(jobs: &Jobs, spec: SimplifySpec) -> serde_json::Value {
    let queued = jobs.submit_simplify(spec).unwrap();
    let done = finish(jobs, queued.job_id, Duration::from_secs(120));
    assert_eq!(done.phase, Phase::Done, "{:?}", done.error);
    jobs.report_result(queued.job_id).unwrap()
}

fn locks_of(result: &serde_json::Value) -> Vec<Lock> {
    serde_json::from_value(result["locks"].clone()).expect("locks are Spot::locks-shaped")
}

#[test]
fn the_headline_composition_solve_simplify_resolve_measures_the_cost() {
    let jobs = Jobs::new(Arc::new(Silent));
    let mut source_spot = river("QQ+,AKs,76s", "QQ+,AKs,76s", 400);
    // Solve to genuine convergence so "the exploited simplification cannot gain" has a tight
    // baseline to be measured against.
    source_spot.stop.target_exploitability = None;
    source_spot.stop.target_exploitability_pct = 0.25;
    let done = solve(&jobs, source_spot);
    let source = done.job_id;
    let source_ev = done.ev.unwrap();

    // Simplify OOP onto halves, submitted through the wire so the command shape is exercised.
    let command: pkwiz_solver::Command = serde_json::from_str(&format!(
        r#"{{"cmd":"simplify","simplify":{{"jobId":{source},"player":0,"grid":2,
            "purifyThreshold":0.8,"maxBoardCards":3}}}}"#
    ))
    .unwrap();
    let submitted = pkwiz_solver::execute(&jobs, command).unwrap();
    assert_eq!(submitted["kind"], "simplify");
    assert_eq!(submitted["phase"], "queued");
    assert_eq!(submitted["analysis"]["sourceJobId"], source);
    let simplify_id = submitted["jobId"].as_u64().unwrap();
    let status = finish(&jobs, simplify_id, Duration::from_secs(120));
    assert_eq!(status.phase, Phase::Done, "{:?}", status.error);

    // `reportResult` serves it, and its identity and census hold together.
    let result = jobs.report_result(simplify_id).unwrap();
    assert_eq!(result["formatVersion"], 1);
    assert_eq!(result["sourceJobId"], source);
    assert_eq!(result["player"], 0);
    assert_eq!(result["grid"], 2);
    assert_eq!(
        result["truncated"], false,
        "a river tree has no streets to cut"
    );
    let locks = locks_of(&result);
    assert!(!locks.is_empty(), "OOP has decision nodes to lock");
    assert_eq!(result["nodesLocked"].as_u64(), Some(locks.len() as u64));
    assert_eq!(
        status.analysis.unwrap().rows,
        Some(locks.len() as u64),
        "the terminal frame's rows are the locked-node count"
    );

    // Every emitted column is on the grid: entries in {0, ½, 1}, sums exactly 1 — or all-zero
    // for a hand left free.
    for (i, lock) in locks.iter().enumerate() {
        let hands = lock.strategy[0].len();
        for h in 0..hands {
            let column: Vec<f32> = lock.strategy.iter().map(|row| row[h]).collect();
            for &v in &column {
                assert!(
                    v == 0.0 || (v - 0.5).abs() < 1e-6 || (v - 1.0).abs() < 1e-6,
                    "lock {i} hand {h}: {v} is off-grid"
                );
            }
            let sum: f64 = column.iter().map(|&v| f64::from(v)).sum();
            assert!(
                sum == 0.0 || (sum - 1.0).abs() < 1e-6,
                "lock {i} hand {h}: sum {sum}"
            );
        }
    }

    // The measurement: resolve with the locks, opponent free to exploit, converged tight.
    let stop = Stop {
        max_iterations: 1000,
        target_exploitability: None,
        target_exploitability_pct: 0.1,
        check_interval: 10,
    };
    let queued = jobs
        .resolve(source, Some(locks.clone()), Some(stop), None, None, None)
        .unwrap();
    let resolved = finish(&jobs, queued.job_id, Duration::from_secs(300));
    assert_eq!(resolved.phase, Phase::Done, "{:?}", resolved.error);
    assert_eq!(resolved.resolved_from, Some(source));

    // An exploited simplified strategy cannot beat the equilibrium it rounded; the epsilon
    // covers the two runs' residual non-convergence.
    let resolved_ev = resolved.ev.unwrap();
    assert!(
        resolved_ev[0] <= source_ev[0] + 1.0,
        "simplified OOP gained EV against an exploiter: {} vs {}",
        resolved_ev[0],
        source_ev[0]
    );

    // Every locked node reads back pinned, at exactly the lock's values (the columns sum to
    // one, so the engine's per-hand normalization moved nothing).
    for lock in &locks {
        let view = jobs.node(queued.job_id, &lock.history).unwrap();
        assert!(view.is_locked, "history {:?}", lock.history);
        for h in 0..lock.strategy[0].len() {
            if lock.strategy.iter().all(|row| row[h] == 0.0) {
                continue; // A free hand: the solver owned its column.
            }
            for (a, row) in lock.strategy.iter().enumerate() {
                assert!(
                    (view.strategy[a][h] - row[h]).abs() < 1e-5,
                    "history {:?} hand {h} action {a}: {} vs {}",
                    lock.history,
                    view.strategy[a][h],
                    row[h]
                );
            }
        }
    }
}

#[test]
fn purify_fires_on_the_original_probabilities_and_only_when_asked() {
    let jobs = Jobs::new(Arc::new(Silent));
    let source = solve(&jobs, river("QQ+,AKs,76s", "QQ+,AKs,76s", 100)).job_id;

    // Disabled: nothing is ever counted as purified, and quarters survive the grid.
    let mut spec = simplify_spec(source, 0);
    spec.grid = 4;
    spec.purify_threshold = None;
    let plain = simplify(&jobs, spec);
    assert_eq!(plain["handsPurified"].as_u64(), Some(0));
    let has_fraction = locks_of(&plain).iter().any(|lock| {
        lock.strategy
            .iter()
            .flatten()
            .any(|&v| v > 0.0 && (v - 1.0).abs() > 1e-6)
    });
    assert!(
        has_fraction,
        "grid 4 alone must not purify a partially mixed strategy"
    );

    // Enabled: hands whose *original* top frequency meets the threshold come out pure.
    let mut spec = simplify_spec(source, 0);
    spec.grid = 4;
    spec.purify_threshold = Some(0.8);
    let purified = simplify(&jobs, spec);
    assert!(purified["handsPurified"].as_u64().unwrap() > 0);
    for lock in locks_of(&purified) {
        let view = jobs.node(source, &lock.history).unwrap();
        for h in 0..view.hands.len() {
            if view.weights[h] == 0.0 {
                continue;
            }
            let top = view
                .strategy
                .iter()
                .map(|row| row[h])
                .fold(f32::NEG_INFINITY, f32::max);
            if top >= 0.8 {
                let column: Vec<f32> = lock.strategy.iter().map(|row| row[h]).collect();
                assert_eq!(
                    column.iter().filter(|&&v| v == 1.0).count(),
                    1,
                    "history {:?} hand {h}: {column:?} should be pure",
                    lock.history
                );
                assert!(column.iter().all(|&v| v == 0.0 || v == 1.0));
            }
        }
    }
}

#[test]
fn grid_one_purifies_every_locked_column() {
    let jobs = Jobs::new(Arc::new(Silent));
    let source = solve(&jobs, river("QQ+,AKs,76s", "QQ+,AKs,76s", 64)).job_id;

    let mut spec = simplify_spec(source, 0);
    spec.grid = 1;
    spec.purify_threshold = None;
    let result = simplify(&jobs, spec);
    for (i, lock) in locks_of(&result).iter().enumerate() {
        for h in 0..lock.strategy[0].len() {
            let column: Vec<f32> = lock.strategy.iter().map(|row| row[h]).collect();
            let ones = column.iter().filter(|&&v| v == 1.0).count();
            let zeros = column.iter().filter(|&&v| v == 0.0).count();
            assert_eq!(ones + zeros, column.len(), "lock {i} hand {h}: {column:?}");
            assert!(ones <= 1, "lock {i} hand {h}: {column:?}");
        }
    }
}

#[test]
fn zero_weight_hands_stay_free_and_the_hands_guard_survives_a_resolve() {
    // A polar spot: AA is pure value, 65s pure air, so grid-1 purification splits the root
    // between actions and every deeper OOP node has hands whose own reach is exactly zero.
    let jobs = Jobs::new(Arc::new(Silent));
    let source = solve(&jobs, river("AA,65s", "KK", 200)).job_id;

    let mut spec = simplify_spec(source, 0);
    spec.grid = 1;
    spec.purify_threshold = None;
    let pure = locks_of(&simplify(&jobs, spec));
    let queued = jobs
        .resolve(source, Some(pure), None, None, None, None)
        .unwrap();
    let resolved = finish(&jobs, queued.job_id, Duration::from_secs(120));
    assert_eq!(resolved.phase, Phase::Done, "{:?}", resolved.error);
    let resolved_id = queued.job_id;

    // Simplify the purified job: below the root, the hands pinned onto the other action are
    // unreachable — zero normalized weight — and must come out free, not frozen.
    let mut spec = simplify_spec(resolved_id, 0);
    spec.include_hands = true;
    let result = simplify(&jobs, spec);
    assert!(
        result["handsFree"].as_u64().unwrap() > 0,
        "a purified strategy leaves zero-reach hands below every action it never takes"
    );
    let locks = locks_of(&result);
    for lock in &locks {
        let view = jobs.node(resolved_id, &lock.history).unwrap();
        // The guard matches the node's hand list entry for entry.
        assert_eq!(lock.hands.as_ref(), Some(&view.hands));
        for h in 0..view.hands.len() {
            if view.weights[h] == 0.0 {
                assert!(
                    lock.strategy.iter().all(|row| row[h] == 0.0),
                    "history {:?} hand {h} has zero weight but a pinned column",
                    lock.history
                );
            } else {
                let sum: f64 = lock.strategy.iter().map(|row| f64::from(row[h])).sum();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "history {:?} hand {h}",
                    lock.history
                );
            }
        }
    }

    // The guards then pass `apply_locks`' hand check inside a resolve of the resolved job.
    let queued = jobs
        .resolve(resolved_id, Some(locks), None, None, None, None)
        .unwrap();
    let guarded = finish(&jobs, queued.job_id, Duration::from_secs(120));
    assert_eq!(guarded.phase, Phase::Done, "{:?}", guarded.error);
}

#[test]
fn only_the_requested_players_nodes_are_locked() {
    let jobs = Jobs::new(Arc::new(Silent));
    let source = solve(&jobs, river("QQ+,AKs", "QQ+,AKs", 32)).job_id;

    for player in 0..2usize {
        let result = simplify(&jobs, simplify_spec(source, player));
        let locks = locks_of(&result);
        assert!(!locks.is_empty(), "player {player} acts somewhere");
        for lock in locks {
            let view = jobs.node(source, &lock.history).unwrap();
            assert_eq!(
                view.player,
                Some(player),
                "history {:?} is not player {player}'s node",
                lock.history
            );
        }
    }
}

#[test]
fn the_street_bound_truncates_a_flop_source_at_three_cards() {
    let jobs = Jobs::new(Arc::new(Silent));
    let source = solve(&jobs, spot("Td9d6h", "QQ+", "QQ+", 8)).job_id;

    let result = simplify(&jobs, simplify_spec(source, 0));
    assert_eq!(result["maxBoardCards"], 3);
    assert_eq!(result["truncated"], true, "the turn was never descended");
    let locks = locks_of(&result);
    assert!(!locks.is_empty());
    for lock in locks {
        let view = jobs.node(source, &lock.history).unwrap();
        assert_eq!(view.board.len(), 3, "history {:?}", lock.history);
    }
}

#[test]
fn cancel_publishes_nothing_and_the_refusals_are_stable() {
    // The deep tree from the dump-cancel test: slow enough that a cancel lands mid-walk.
    let collector = Arc::new(Collector::default());
    let jobs = Jobs::with_throttle(Arc::clone(&collector) as Arc<dyn Emit>, Duration::ZERO);
    let mut deep = spot(
        "Td9d6h",
        "88+,AQs+,AKo,KQs,JTs,T9s",
        "88+,AQs+,AKo,KQs,JTs,T9s",
        2,
    );
    deep.effective_stack = 900;
    let source = solve(&jobs, deep).job_id;

    let mut spec = simplify_spec(source, 0);
    spec.max_board_cards = 5;
    let simplify_id = jobs.submit_simplify(spec).unwrap().job_id;
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let status = jobs.status(simplify_id).unwrap();
        if status.phase == Phase::Running && status.analysis.as_ref().is_some_and(|a| a.nodes > 0) {
            break;
        }
        assert!(Instant::now() < deadline, "the simplify never got going");
        std::thread::sleep(Duration::from_millis(2));
    }
    jobs.cancel(simplify_id).unwrap();
    let cancelled = finish(&jobs, simplify_id, Duration::from_secs(30));
    assert_eq!(cancelled.phase, Phase::Cancelled);
    assert_eq!(cancelled.stopped, Some(Stopped::Cancelled));

    // Nothing was published: the report mirror.
    let err = jobs.report_result(simplify_id).unwrap_err();
    assert_eq!(err.code(), "not_readable");

    // The terminal frame is the job's last, and the session is alive.
    let frames: Vec<serde_json::Value> = collector
        .events()
        .into_iter()
        .filter(|e| e["job"]["jobId"].as_u64() == Some(simplify_id))
        .collect();
    assert_eq!(frames.last().unwrap()["job"]["phase"], "cancelled");
    assert!(jobs.node(source, &[]).is_ok());

    // `reportResult` on a non-report, non-simplify job stays `not_report`.
    assert_eq!(jobs.report_result(source).unwrap_err().code(), "not_report");

    // A simplify naming a source with no tree is refused synchronously, like any analysis.
    let prep: pkwiz_solver::BunchingSpec =
        serde_json::from_str(r#"{"foldRanges":["AA"],"flop":"Td9d6h"}"#).unwrap();
    let prep_id = jobs.submit_bunching(prep).unwrap().job_id;
    let err = jobs.submit_simplify(simplify_spec(prep_id, 0)).unwrap_err();
    assert_eq!(err.code(), "not_readable");
    assert!(err.to_string().contains("bunching"), "{err}");
    jobs.cancel(prep_id).unwrap();
    let err = jobs
        .submit_simplify(simplify_spec(simplify_id, 0))
        .unwrap_err();
    assert_eq!(err.code(), "not_readable");
    assert!(err.to_string().contains("simplify"), "{err}");
    assert_eq!(
        jobs.submit_simplify(simplify_spec(9999, 0))
            .unwrap_err()
            .code(),
        "no_such_job"
    );
}

#[test]
fn a_bad_spec_fails_the_command_not_a_job() {
    let jobs = Jobs::new(Arc::new(Silent));
    let source = solve(&jobs, river("QQ+", "QQ+", 8)).job_id;

    let mut bad_player = simplify_spec(source, 2);
    bad_player.player = 2;
    assert_eq!(
        jobs.submit_simplify(bad_player).unwrap_err().code(),
        "bad_player"
    );

    for (mutate, needle) in [
        (
            Box::new(|s: &mut SimplifySpec| s.grid = 0) as Box<dyn Fn(&mut SimplifySpec)>,
            "simplify.grid",
        ),
        (
            Box::new(|s: &mut SimplifySpec| s.grid = 65),
            "simplify.grid",
        ),
        (
            Box::new(|s: &mut SimplifySpec| s.max_board_cards = 6),
            "simplify.maxBoardCards",
        ),
        (
            Box::new(|s: &mut SimplifySpec| s.purify_threshold = Some(0.3)),
            "simplify.purifyThreshold",
        ),
        (
            Box::new(|s: &mut SimplifySpec| s.purify_threshold = Some(1.5)),
            "simplify.purifyThreshold",
        ),
    ] {
        let mut spec = simplify_spec(source, 0);
        mutate(&mut spec);
        let err = jobs.submit_simplify(spec).unwrap_err();
        assert_eq!(err.code(), "engine");
        assert!(err.to_string().contains(needle), "{err}");
    }

    let mut bad_history = simplify_spec(source, 0);
    bad_history.history = vec![usize::MAX];
    assert_eq!(
        jobs.submit_simplify(bad_history).unwrap_err().code(),
        "bad_node"
    );

    // And the wire shape fills its documented defaults.
    let command: pkwiz_solver::Command =
        serde_json::from_str(r#"{"cmd":"simplify","simplify":{"jobId":1,"player":1}}"#).unwrap();
    let pkwiz_solver::Command::Simplify { simplify } = command else {
        panic!("parsed as something else");
    };
    assert_eq!(simplify.player, 1);
    assert_eq!(simplify.grid, 2);
    assert_eq!(simplify.max_board_cards, 3);
    assert_eq!(simplify.purify_threshold, Some(0.8));
    assert!(!simplify.include_hands);
    assert!(simplify.history.is_empty());
}

#[test]
fn an_opponent_only_region_yields_an_empty_locks_array_not_an_error() {
    let jobs = Jobs::new(Arc::new(Silent));
    let source = solve(&jobs, river("QQ+,AKs", "QQ+,AKs", 16)).job_id;

    // A base whose subtree holds no decision node of the requested player at all: the last
    // decision on the most aggressive line — under it, the *other* player never acts again.
    let mut history = Vec::new();
    loop {
        let view = jobs.node(source, &history).unwrap();
        if view.is_terminal {
            break;
        }
        // Take the highest action index (the most aggressive line) until the tree ends.
        history.push(view.actions.len() - 1);
    }
    // The node just above the terminal is someone's fold/call decision; under it, the *other*
    // player never acts again.
    let last_decision = history[..history.len() - 1].to_vec();
    let decision_view = jobs.node(source, &last_decision).unwrap();
    let absent_player = 1 - decision_view.player.unwrap();

    let mut spec = simplify_spec(source, absent_player);
    spec.history = last_decision;
    let result = simplify(&jobs, spec);
    assert_eq!(result["nodesLocked"].as_u64(), Some(0));
    assert!(locks_of(&result).is_empty());
    assert!(result["nodesVisited"].as_u64().unwrap() > 0);
}
