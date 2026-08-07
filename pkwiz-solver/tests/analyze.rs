//! Aggregation reports and the full-tree dump.
//!
//! The claim both features make is *consistency*: every number a report or a dump emits must
//! equal what the `node` command answers at the same node — same formulas, same engine reads.
//! So the tests here recompute rows from `NodeView`s and compare, rather than asserting golden
//! values that would break with every solver improvement. The rest is contract: row counts
//! match dealable cards, frequencies sum to one, the dump's header/summary framing marks
//! completeness, `maxBytes` aborts, cancel leaves a partial file and a live session.
//!
//! Every spot is tiny (river or turn boards, QQ+-sized ranges, ≤ 32 iterations) except the one
//! the cancel test needs to be mid-flight long enough to cancel.

use std::sync::Arc;
use std::time::{Duration, Instant};

use pkwiz_solver::spot::{BoardSpec, RangeSpec, Sizing, StreetSizing};
use pkwiz_solver::{
    Collector, DumpInclude, DumpSpec, Emit, JobStatus, Jobs, Phase, ReportKind, ReportSpec, Silent,
    Spot, Stop, Stopped,
};

/// A spot on any street, sized 50%/2.5x on every present street — small with QQ+-class ranges.
fn spot(board: &str, range: &str, iterations: u32) -> Spot {
    Spot {
        oop: RangeSpec::Notation(range.to_owned()),
        ip: RangeSpec::Notation(range.to_owned()),
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
fn river(board: &str, range: &str, iterations: u32) -> Spot {
    let mut spot = spot(board, range, iterations);
    spot.sizing.flop = StreetSizing::none();
    spot.sizing.turn = StreetSizing::none();
    spot
}

fn report_spec(job_id: u64, kind: ReportKind) -> ReportSpec {
    ReportSpec {
        job_id,
        history: Vec::new(),
        kind,
        line: Vec::new(),
        categories: false,
    }
}

fn dump_spec(job_id: u64, path: &std::path::Path) -> DumpSpec {
    DumpSpec {
        job_id,
        path: path.to_string_lossy().into_owned(),
        history: Vec::new(),
        max_board_cards: 5,
        include: DumpInclude::default(),
        max_bytes: None,
        compress: false,
        compression_level: Spot::default_compression_level(),
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
    let dir = std::env::temp_dir().join(format!("pkwiz-analyze-{}-{label}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn solve(spot: Spot) -> (Jobs, JobStatus) {
    let jobs = Jobs::new(Arc::new(Silent));
    let queued = jobs.submit(spot).expect("the spot is valid");
    let done = finish(&jobs, queued.job_id, Duration::from_secs(120));
    assert_eq!(done.phase, Phase::Done, "{:?}", done.error);
    (jobs, done)
}

fn card_id(s: &str) -> usize {
    pkwiz_range::Card::parse(s).unwrap().index() as usize
}

/// `Σ v·w` in f64 — the report formulas, recomputed independently.
fn wsum(values: &[f32], weights: &[f32]) -> f64 {
    values
        .iter()
        .zip(weights)
        .map(|(&v, &w)| f64::from(v) * f64::from(w))
        .sum()
}

#[test]
fn a_runouts_report_covers_every_dealable_river_and_matches_the_node_command() {
    // A turn spot; check-check reaches the river chance node.
    let (jobs, done) = solve(spot("Td9d6hQc", "QQ+", 32));
    let source = done.job_id;

    let chance = jobs.node(source, &[0, 0]).unwrap();
    assert!(chance.is_chance);
    let dealable = chance.actions.len();

    let mut spec = report_spec(source, ReportKind::Runouts);
    spec.history = vec![0, 0];
    spec.categories = true;
    let queued = jobs.submit_report(spec).unwrap();
    let status = finish(&jobs, queued.job_id, Duration::from_secs(60));
    assert_eq!(status.phase, Phase::Done, "{:?}", status.error);

    // The status carries the analysis identity and the row count.
    let analysis = status.analysis.expect("a report job has analysis");
    assert_eq!(analysis.source_job_id, source);
    assert_eq!(analysis.report_kind, Some(ReportKind::Runouts));
    assert_eq!(analysis.rows, Some(dealable as u64));

    let result = jobs.report_result(queued.job_id).unwrap();
    assert_eq!(result["formatVersion"], 1);
    assert_eq!(result["kind"], "runouts");
    assert_eq!(result["sourceJobId"], source);
    assert_eq!(result["street"], "river");
    let rows = result["rows"].as_array().unwrap();
    assert_eq!(rows.len(), dealable, "one row per dealable river");

    // Frequencies sum to one on every live row, and every row recomputes from the node
    // command bit-for-bit up to f64 summation.
    let player = result["player"].as_u64().unwrap() as usize;
    for (i, sample) in [0, rows.len() / 2, rows.len() - 1].iter().enumerate() {
        let row = &rows[*sample];
        let card = row["card"].as_str().unwrap();
        let view = jobs.node(source, &[0, 0, card_id(card)]).unwrap();
        assert_eq!(view.player, Some(player), "sample {i}");

        let matchups: f64 = view.weights.iter().map(|&w| f64::from(w)).sum();
        assert!(
            (row["matchups"].as_f64().unwrap() - matchups).abs() < 1e-3 * matchups.max(1.0),
            "sample {i}: {} vs {matchups}",
            row["matchups"]
        );

        let frequencies = row["frequencies"].as_array().unwrap();
        assert_eq!(frequencies.len(), view.actions.len());
        let sum: f64 = frequencies.iter().map(|f| f.as_f64().unwrap()).sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "sample {i}: frequencies sum {sum}"
        );
        for (a, frequency) in frequencies.iter().enumerate() {
            let expected = wsum(&view.strategy[a], &view.weights) / matchups;
            assert!(
                (frequency.as_f64().unwrap() - expected).abs() < 1e-6,
                "sample {i} action {a}"
            );
        }

        // The acting player's average equity/EV recompute from the same view.
        let eq = wsum(&view.equity, &view.weights) / matchups;
        let ev = wsum(&view.ev, &view.weights) / matchups;
        let row_eq = row["averageEquity"][player].as_f64().unwrap();
        let row_ev = row["averageEv"][player].as_f64().unwrap();
        assert!((row_eq - eq).abs() < 1e-4, "sample {i}: {row_eq} vs {eq}");
        assert!(
            (row_ev - ev).abs() < 1e-3 * ev.abs().max(1.0),
            "sample {i}: {row_ev} vs {ev}"
        );

        // Categories: a stable 14-row set partitioning the row's matchups.
        let categories = row["categories"].as_array().unwrap();
        assert_eq!(categories.len(), 14, "every category, empty ones included");
        let total: f64 = categories
            .iter()
            .map(|c| c["matchups"].as_f64().unwrap())
            .sum();
        assert!(
            (total - matchups).abs() < 1e-3 * matchups.max(1.0),
            "sample {i}: categories hold {total} of {matchups}"
        );
        for category in categories {
            if category["matchups"].as_f64().unwrap() == 0.0 {
                assert!(category["frequencies"].is_null());
                assert!(category["averageEquity"].is_null());
                assert!(category["draws"].is_null());
            } else {
                assert!(category["draws"]["flushDraw"].is_number());
            }
        }
    }
}

#[test]
fn a_lines_report_walks_the_street_with_reach_algebra() {
    let (jobs, done) = solve(river("2c7dTh4sQd", "QQ+,AKs,76s", 32));
    let source = done.job_id;

    let queued = jobs
        .submit_report(report_spec(source, ReportKind::Lines))
        .unwrap();
    let status = finish(&jobs, queued.job_id, Duration::from_secs(60));
    assert_eq!(status.phase, Phase::Done, "{:?}", status.error);

    let result = jobs.report_result(queued.job_id).unwrap();
    assert_eq!(result["kind"], "lines");
    let rows = result["rows"].as_array().unwrap();
    assert!(rows.len() >= 3, "a river tree has several decision nodes");

    // The base row leads, at full reach.
    assert_eq!(rows[0]["line"].as_array().unwrap().len(), 0);
    assert_eq!(rows[0]["lineText"], "(root)");
    assert_eq!(rows[0]["reach"].as_f64().unwrap(), 1.0);

    // Depth-first in action order: the second row is the base's first action.
    assert_eq!(rows[1]["line"][0], 0);

    let base_matchups = rows[0]["matchups"].as_f64().unwrap();
    for row in rows {
        let matchups = row["matchups"].as_f64().unwrap();
        if matchups > 0.0 {
            let sum: f64 = row["frequencies"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| f.as_f64().unwrap())
                .sum();
            assert!((sum - 1.0).abs() < 1e-4, "{}: {sum}", row["lineText"]);
        }

        // Reach algebra: a child's matchups are its parent's, thinned by the frequency of the
        // action that reaches it.
        let line: Vec<usize> = row["line"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        if let Some((&last, parent_line)) = line.split_last() {
            let parent = rows
                .iter()
                .find(|r| {
                    r["line"].as_array().unwrap().len() == parent_line.len()
                        && r["line"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .zip(parent_line)
                            .all(|(v, &p)| v.as_u64().unwrap() as usize == p)
                })
                .expect("every non-base row's parent is a row");
            let expected = parent["matchups"].as_f64().unwrap()
                * parent["frequencies"][last].as_f64().unwrap();
            assert!(
                (matchups - expected).abs() < 1e-3 * expected.max(1.0),
                "{}: {matchups} vs parent × frequency {expected}",
                row["lineText"]
            );
            assert!(
                (row["reach"].as_f64().unwrap() - matchups / base_matchups).abs() < 1e-6,
                "{}",
                row["lineText"]
            );
        }
    }
}

#[test]
fn a_categories_report_partitions_the_range_exactly_once() {
    let (jobs, done) = solve(river("2c7dTh4sQd", "QQ+,AKs,76s", 32));
    let source = done.job_id;

    let queued = jobs
        .submit_report(report_spec(source, ReportKind::Categories))
        .unwrap();
    let status = finish(&jobs, queued.job_id, Duration::from_secs(60));
    assert_eq!(status.phase, Phase::Done, "{:?}", status.error);

    let result = jobs.report_result(queued.job_id).unwrap();
    assert_eq!(result["kind"], "categories");
    let view = jobs.node(source, &[]).unwrap();
    assert_eq!(result["player"].as_u64(), Some(0));

    let node_matchups: f64 = view.weights.iter().map(|&w| f64::from(w)).sum();
    let rows = result["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 14);

    let total: f64 = rows.iter().map(|r| r["matchups"].as_f64().unwrap()).sum();
    assert!(
        (total - node_matchups).abs() < 1e-3 * node_matchups,
        "{total} vs {node_matchups}"
    );
    let pct: f64 = rows.iter().map(|r| r["rangePct"].as_f64().unwrap()).sum();
    assert!((pct - 100.0).abs() < 1e-6, "{pct}");

    // Every hand appears in exactly one category row.
    let mut listed: Vec<String> = rows
        .iter()
        .flat_map(|r| {
            r["hands"]
                .as_array()
                .map(|hands| {
                    hands
                        .iter()
                        .map(|h| h.as_str().unwrap().to_owned())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        })
        .collect();
    let mut expected = view.hands.clone();
    listed.sort();
    expected.sort();
    assert_eq!(listed, expected);
}

/// Walk the solved tree through the `node` command, counting nodes the way the dump does.
fn count_nodes(jobs: &Jobs, id: u64, history: &mut Vec<usize>, counts: &mut [u64; 3]) {
    let view = jobs.node(id, history).unwrap();
    if view.is_terminal {
        counts[2] += 1;
        return;
    }
    if view.is_chance {
        counts[1] += 1;
        for action in &view.actions {
            let card = action.trim_start_matches("Chance(").trim_end_matches(')');
            history.push(card_id(card));
            count_nodes(jobs, id, history, counts);
            history.pop();
        }
        return;
    }
    counts[0] += 1;
    for index in 0..view.actions.len() {
        history.push(index);
        count_nodes(jobs, id, history, counts);
        history.pop();
    }
}

#[test]
fn a_dump_round_trips_against_the_node_command() {
    // A turn board whose four suits appear in distinct patterns, so no suit isomorphism
    // dedupes the river deals and the independent traversal counts exactly what the dump
    // writes.
    let dir = temp_dir("round-trip");
    let path = dir.join("turn.jsonl");
    let (jobs, done) = solve(spot("Td9d6hQc", "QQ+", 16));
    let source = done.job_id;

    let queued = jobs.submit_dump(dump_spec(source, &path)).unwrap();
    let status = finish(&jobs, queued.job_id, Duration::from_secs(120));
    assert_eq!(status.phase, Phase::Done, "{:?}", status.error);
    assert_eq!(status.saved_to.as_deref(), Some(&*path.to_string_lossy()));

    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).expect("every line is one JSON object"))
        .collect();

    // Framing: header first, summary last, nothing typed "terminal" anywhere.
    let header = &lines[0];
    assert_eq!(header["t"], "header");
    assert_eq!(header["formatVersion"], 1);
    assert_eq!(header["engineRev"], pkwiz_solver::protocol::ENGINE_REV);
    assert_eq!(header["sourceJobId"], source);
    assert_eq!(header["board"].as_array().unwrap().len(), 4);
    let summary = lines.last().unwrap();
    assert_eq!(summary["t"], "summary");
    assert_eq!(summary["complete"], true);
    assert_eq!(summary["truncated"], false);
    assert!(lines.iter().all(|l| l["t"] != "terminal"));

    // The line census equals an independent traversal through the node command.
    let mut counts = [0u64; 3];
    count_nodes(&jobs, source, &mut Vec::new(), &mut counts);
    let node_lines: Vec<&serde_json::Value> = lines.iter().filter(|l| l["t"] == "node").collect();
    let chance_lines: Vec<&serde_json::Value> =
        lines.iter().filter(|l| l["t"] == "chance").collect();
    assert_eq!(node_lines.len() as u64, counts[0], "decision nodes");
    assert_eq!(chance_lines.len() as u64, counts[1], "chance nodes");
    assert_eq!(summary["decisionNodes"].as_u64(), Some(counts[0]));
    assert_eq!(summary["chanceNodes"].as_u64(), Some(counts[1]));
    assert_eq!(summary["terminalNodes"].as_u64(), Some(counts[2]));
    assert_eq!(
        summary["nodes"].as_u64(),
        Some(counts.iter().sum::<u64>()),
        "the summary counts terminals it never wrote"
    );

    // Sampled node lines reproduce the node command bit-for-bit.
    let step = (node_lines.len() / 5).max(1);
    for line in node_lines.iter().step_by(step).take(5) {
        let history: Vec<usize> = line["history"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let view = jobs.node(source, &history).unwrap();
        assert_eq!(line["player"].as_u64(), view.player.map(|p| p as u64));
        let strategy: Vec<Vec<f32>> = serde_json::from_value(line["strategy"].clone()).unwrap();
        assert_eq!(strategy, view.strategy, "history {history:?}");
        let actions: Vec<String> = serde_json::from_value(line["actions"].clone()).unwrap();
        assert_eq!(actions, view.actions);
        assert_eq!(line["isLocked"], view.is_locked);
        // Default include: strategy only.
        assert!(line["ev"].is_null() && line["equity"].is_null() && line["weights"].is_null());
    }

    // Chance lines carry every dealable card, matching `possible_cards` via the node command.
    for line in &chance_lines {
        let history: Vec<usize> = line["history"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let view = jobs.node(source, &history).unwrap();
        let expected: Vec<String> = view
            .actions
            .iter()
            .map(|a| {
                a.trim_start_matches("Chance(")
                    .trim_end_matches(')')
                    .to_owned()
            })
            .collect();
        let cards: Vec<String> = serde_json::from_value(line["cards"].clone()).unwrap();
        assert_eq!(cards, expected, "history {history:?}");
        assert_eq!(line["street"], "river");
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_dump_stops_at_the_street_bound() {
    let dir = temp_dir("street-bound");
    let path = dir.join("flop-only.jsonl");
    let (jobs, done) = solve(spot("Td9d6h", "QQ+", 8));

    let mut spec = dump_spec(done.job_id, &path);
    spec.max_board_cards = 3;
    let queued = jobs.submit_dump(spec).unwrap();
    let status = finish(&jobs, queued.job_id, Duration::from_secs(60));
    assert_eq!(status.phase, Phase::Done, "{:?}", status.error);

    let lines: Vec<serde_json::Value> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.last().unwrap()["truncated"], true);

    // Every chance line is childless: no node line's history extends one.
    let chance_histories: Vec<Vec<u64>> = lines
        .iter()
        .filter(|l| l["t"] == "chance")
        .map(|l| {
            l["history"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect()
        })
        .collect();
    assert!(!chance_histories.is_empty(), "a flop tree has chance nodes");
    for line in lines.iter().filter(|l| l["t"] == "node") {
        let history: Vec<u64> = line["history"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert!(
            !chance_histories
                .iter()
                .any(|ch| history.len() > ch.len() && history[..ch.len()] == ch[..]),
            "node {history:?} descends past a chance node despite maxBoardCards 3"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_dump_past_max_bytes_fails_and_leaves_the_partial_file() {
    let dir = temp_dir("max-bytes");
    let path = dir.join("aborted.jsonl");
    let (jobs, done) = solve(river("2c7dTh4sQd", "QQ+,AKs", 16));

    let mut spec = dump_spec(done.job_id, &path);
    spec.max_bytes = Some(600);
    let queued = jobs.submit_dump(spec).unwrap();
    let status = finish(&jobs, queued.job_id, Duration::from_secs(60));
    assert_eq!(status.phase, Phase::Failed);
    let error = status.error.expect("a failure says why");
    assert!(
        error.contains("maxBytes") && error.contains("600"),
        "{error}"
    );

    // The partial file is left, and its missing summary line marks it incomplete.
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(!text.contains(r#""t":"summary""#));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_compressed_dump_decodes_to_the_uncompressed_bytes() {
    let dir = temp_dir("zstd");
    let raw = dir.join("raw.jsonl");
    let packed = dir.join("packed.jsonl.zst");
    let (jobs, done) = solve(river("2c7dTh4sQd", "QQ+,AKs", 16));

    let queued = jobs.submit_dump(dump_spec(done.job_id, &raw)).unwrap();
    assert_eq!(
        finish(&jobs, queued.job_id, Duration::from_secs(60)).phase,
        Phase::Done
    );

    let mut spec = dump_spec(done.job_id, &packed);
    spec.compress = true;
    let queued = jobs.submit_dump(spec).unwrap();
    let status = finish(&jobs, queued.job_id, Duration::from_secs(60));
    assert_eq!(status.phase, Phase::Done, "{:?}", status.error);

    let raw_bytes = std::fs::read(&raw).unwrap();
    let packed_bytes = std::fs::read(&packed).unwrap();
    assert!(
        packed_bytes.len() < raw_bytes.len(),
        "compression compresses"
    );
    let decoded = zstd::decode_all(&packed_bytes[..]).unwrap();
    assert_eq!(
        decoded, raw_bytes,
        "one deterministic traversal, two framings"
    );

    // The status reports the file's real (compressed) size.
    let analysis = status.analysis.unwrap();
    assert_eq!(analysis.bytes_written, Some(packed_bytes.len() as u64));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn cancelling_a_dump_leaves_a_partial_file_and_a_live_session() {
    // A deep-stacked full flop tree with every array included takes seconds to stream — and
    // the cancel lands milliseconds after the first node, so the terminal phase is
    // deterministic: the dump cannot finish before the flag is seen.
    let dir = temp_dir("cancel");
    let path = dir.join("cancelled.jsonl");
    let collector = Arc::new(Collector::default());
    let jobs = Jobs::with_throttle(Arc::clone(&collector) as Arc<dyn Emit>, Duration::ZERO);

    let mut deep = spot("Td9d6h", "88+,AQs+,AKo,KQs,JTs,T9s", 2);
    deep.effective_stack = 900;
    let queued = jobs.submit(deep).unwrap();
    let done = finish(&jobs, queued.job_id, Duration::from_secs(300));
    assert_eq!(done.phase, Phase::Done, "{:?}", done.error);
    let source = done.job_id;

    let mut spec = dump_spec(source, &path);
    spec.include = DumpInclude {
        strategy: true,
        ev: true,
        equity: true,
        weights: true,
    };
    let dump_id = jobs.submit_dump(spec).unwrap().job_id;

    // Wait for genuine progress — a running frame with nodes visited — then cancel.
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let status = jobs.status(dump_id).unwrap();
        if status.phase == Phase::Running && status.analysis.as_ref().is_some_and(|a| a.nodes > 0) {
            break;
        }
        assert!(Instant::now() < deadline, "the dump never got going");
        std::thread::sleep(Duration::from_millis(2));
    }
    jobs.cancel(dump_id).unwrap();
    let cancelled = finish(&jobs, dump_id, Duration::from_secs(30));
    assert_eq!(cancelled.phase, Phase::Cancelled);
    assert_eq!(cancelled.stopped, Some(Stopped::Cancelled));

    // The partial file exists and its missing summary marks it incomplete.
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.starts_with(r#"{"t":"header""#));
    assert!(!text.contains(r#""t":"summary""#));

    // The terminal frame is the last one for this job, and carries the cancel.
    let frames: Vec<serde_json::Value> = collector
        .events()
        .into_iter()
        .filter(|e| e["job"]["jobId"].as_u64() == Some(dump_id))
        .collect();
    assert_eq!(frames.last().unwrap()["job"]["phase"], "cancelled");
    assert_eq!(frames.last().unwrap()["job"]["stopped"], "cancelled");

    // And the session is alive: the source answers node reads and the queue takes work.
    assert!(jobs.node(source, &[]).is_ok());
    assert!(jobs.status(source).is_ok());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_released_source_reports_identically_after_reload() {
    let dir = temp_dir("released-source");
    let path = dir.join("source.bin");

    let mut turn_spot = spot("Td9d6hQc", "QQ+", 16);
    turn_spot.save_path = Some(path.to_string_lossy().into_owned());
    let (jobs, done) = solve(turn_spot);
    let source = done.job_id;

    let mut spec = report_spec(source, ReportKind::Runouts);
    spec.history = vec![0, 0];
    spec.categories = true;

    let first = jobs.submit_report(spec.clone()).unwrap();
    assert_eq!(
        finish(&jobs, first.job_id, Duration::from_secs(60)).phase,
        Phase::Done
    );
    let before = jobs.report_result(first.job_id).unwrap();

    jobs.release(source).unwrap();
    assert!(!jobs.status(source).unwrap().resident);

    // The report's acquire path reloads the file under the source's own cap; the numbers must
    // not move.
    let second = jobs.submit_report(spec).unwrap();
    assert_eq!(
        finish(&jobs, second.job_id, Duration::from_secs(60)).phase,
        Phase::Done
    );
    let mut after = jobs.report_result(second.job_id).unwrap();
    // The result names its own source job — identical here — so compare verbatim.
    assert_eq!(after["sourceJobId"], before["sourceJobId"]);
    after["sourceJobId"] = before["sourceJobId"].clone();
    assert_eq!(after, before);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn report_and_dump_errors_carry_stable_codes() {
    let dir = temp_dir("errors");
    let (jobs, done) = solve(river("2c7dTh4sQd", "QQ+", 16));
    let source = done.job_id;

    // Unknown source.
    let err = jobs
        .submit_report(report_spec(9999, ReportKind::Lines))
        .unwrap_err();
    assert_eq!(err.code(), "no_such_job");

    // A bunching preparation has no tree to aggregate, whatever its phase.
    let spec: pkwiz_solver::BunchingSpec =
        serde_json::from_str(r#"{"foldRanges":["AA"],"flop":"2c7dTh"}"#).unwrap();
    let prep = jobs.submit_bunching(spec).unwrap();
    let err = jobs
        .submit_report(report_spec(prep.job_id, ReportKind::Lines))
        .unwrap_err();
    assert_eq!(err.code(), "not_readable");
    assert!(err.to_string().contains("bunching"), "{err}");
    jobs.cancel(prep.job_id).unwrap();

    // reportResult on a solve job.
    let err = jobs.report_result(source).unwrap_err();
    assert_eq!(err.code(), "not_report");

    // reportResult on a report that is not done yet: park a long solve in front of it so the
    // report is deterministically still queued when asked.
    let blocker = jobs
        .submit(river("2c7dTh4sQd", "QQ+,AKs,76s", 1_000_000))
        .unwrap()
        .job_id;
    let queued = jobs
        .submit_report(report_spec(source, ReportKind::Lines))
        .unwrap();
    let err = jobs.report_result(queued.job_id).unwrap_err();
    assert_eq!(err.code(), "not_readable");
    assert!(err.to_string().contains("once it is done"), "{err}");
    jobs.cancel(blocker).unwrap();
    finish(&jobs, blocker, Duration::from_secs(30));
    finish(&jobs, queued.job_id, Duration::from_secs(60));

    // A runouts base that reaches a decision node fails the job and names the node kind.
    let failed = jobs
        .submit_report(report_spec(source, ReportKind::Runouts))
        .unwrap();
    let status = finish(&jobs, failed.job_id, Duration::from_secs(60));
    assert_eq!(status.phase, Phase::Failed);
    let error = status.error.unwrap();
    assert!(
        error.contains("decision") && error.contains("chance"),
        "{error}"
    );
    // A failed report has no result.
    assert_eq!(
        jobs.report_result(failed.job_id).unwrap_err().code(),
        "not_readable"
    );

    // A dump into a nonexistent directory fails the job with the I/O reason.
    let bad = dump_spec(source, &dir.join("no-such-dir").join("x.jsonl"));
    let failed = jobs.submit_dump(bad).unwrap();
    let status = finish(&jobs, failed.job_id, Duration::from_secs(60));
    assert_eq!(status.phase, Phase::Failed);
    assert!(
        status.error.unwrap().contains("could not write"),
        "io reason"
    );

    // Synchronous refusals: an empty path and an out-of-range street bound.
    let mut empty = dump_spec(source, std::path::Path::new(""));
    empty.path = String::new();
    assert_eq!(jobs.submit_dump(empty).unwrap_err().code(), "engine");
    let mut bad_streets = dump_spec(source, &dir.join("x.jsonl"));
    bad_streets.max_board_cards = 6;
    assert_eq!(jobs.submit_dump(bad_streets).unwrap_err().code(), "engine");

    // `node` and `save` on an analysis job: nothing to read, nothing to save.
    let report = jobs
        .submit_report(report_spec(source, ReportKind::Lines))
        .unwrap();
    finish(&jobs, report.job_id, Duration::from_secs(60));
    assert_eq!(
        jobs.node(report.job_id, &[]).unwrap_err().code(),
        "not_readable"
    );
    assert_eq!(
        jobs.save(report.job_id, "/tmp/x").unwrap_err().code(),
        "nothing_to_save"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_report_on_a_still_queued_source_runs_after_it() {
    // The FIFO allowance: the report is accepted while its source has not even started, and
    // the single worker guarantees the solve finishes first.
    let jobs = Jobs::new(Arc::new(Silent));
    let solve_id = jobs.submit(river("2c7dTh4sQd", "QQ+", 16)).unwrap().job_id;
    let report_id = jobs
        .submit_report(report_spec(solve_id, ReportKind::Lines))
        .unwrap()
        .job_id;

    let status = finish(&jobs, report_id, Duration::from_secs(120));
    assert_eq!(status.phase, Phase::Done, "{:?}", status.error);
    assert!(status.analysis.unwrap().rows.unwrap() > 0);
    assert_eq!(jobs.status(solve_id).unwrap().phase, Phase::Done);
}

#[test]
fn report_frames_carry_kind_and_analysis_and_the_result_round_trips_the_wire() {
    let collector = Arc::new(Collector::default());
    let jobs = Jobs::with_throttle(Arc::clone(&collector) as Arc<dyn Emit>, Duration::ZERO);
    let solve_id = jobs.submit(spot("Td9d6hQc", "QQ+", 16)).unwrap().job_id;
    finish(&jobs, solve_id, Duration::from_secs(120));

    // The report command arrives as wire JSON, exactly as a host writes it.
    let command: pkwiz_solver::Command = serde_json::from_str(&format!(
        r#"{{"cmd":"report","report":{{"jobId":{solve_id},"kind":"runouts","history":[0,0],"categories":true}}}}"#
    ))
    .unwrap();
    let submitted = pkwiz_solver::execute(&jobs, command).unwrap();
    assert_eq!(submitted["kind"], "report");
    assert_eq!(submitted["phase"], "queued");
    assert_eq!(submitted["analysis"]["sourceJobId"], solve_id);
    assert_eq!(submitted["analysis"]["reportKind"], "runouts");
    let report_id = submitted["jobId"].as_u64().unwrap();

    finish(&jobs, report_id, Duration::from_secs(60));

    // Every frame of the report job carries its kind and analysis; the terminal frame has rows.
    let frames: Vec<serde_json::Value> = collector
        .events()
        .into_iter()
        .filter(|e| e["job"]["jobId"].as_u64() == Some(report_id))
        .collect();
    assert!(frames.len() >= 2, "queued and done at minimum");
    assert_eq!(frames[0]["job"]["phase"], "queued");
    for frame in &frames {
        assert_eq!(frame["job"]["kind"], "report");
        assert_eq!(frame["job"]["analysis"]["sourceJobId"], solve_id);
        assert_eq!(frame["job"]["bunching"], serde_json::Value::Null);
    }
    let last = frames.last().unwrap();
    assert_eq!(last["job"]["phase"], "done");
    assert!(last["job"]["analysis"]["rows"].as_u64().unwrap() > 0);

    // reportResult over the wire serves the stored value verbatim.
    let command: pkwiz_solver::Command =
        serde_json::from_str(&format!(r#"{{"cmd":"reportResult","jobId":{report_id}}}"#)).unwrap();
    let served = pkwiz_solver::execute(&jobs, command).unwrap();
    assert_eq!(served, jobs.report_result(report_id).unwrap());
    assert_eq!(served["kind"], "runouts");

    // The dump command parses from wire JSON too, defaults included.
    let command: pkwiz_solver::Command =
        serde_json::from_str(r#"{"cmd":"dump","dump":{"jobId":1,"path":"/tmp/x.jsonl"}}"#).unwrap();
    let pkwiz_solver::Command::Dump { dump } = command else {
        panic!("parsed as something else");
    };
    assert_eq!(dump.max_board_cards, 5);
    assert!(dump.include.strategy && !dump.include.ev);
    assert_eq!(dump.compression_level, Some(3));
}

#[test]
fn a_job_status_without_the_analysis_field_still_deserializes() {
    // The PROTOCOL_VERSION 1 guard: frames captured before `analysis` existed must still parse.
    let (jobs, done) = solve(river("2c7dTh4sQd", "QQ+", 8));
    let mut old = serde_json::to_value(&done).unwrap();
    assert!(old.as_object().unwrap().contains_key("analysis"));
    old.as_object_mut().unwrap().remove("analysis");
    let parsed: JobStatus = serde_json::from_value(old).unwrap();
    assert_eq!(parsed.analysis, None);
    assert_eq!(parsed.job_id, done.job_id);

    // And a report status round-trips through serde with its kind and analysis intact.
    let report = jobs
        .submit_report(report_spec(done.job_id, ReportKind::Lines))
        .unwrap();
    let status = finish(&jobs, report.job_id, Duration::from_secs(60));
    let json = serde_json::to_value(&status).unwrap();
    assert_eq!(json["kind"], "report");
    let back: JobStatus = serde_json::from_value(json).unwrap();
    assert_eq!(back, status);
}
