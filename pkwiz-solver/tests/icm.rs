//! ICM over the wire and around the job lifecycle.
//!
//! The engine's own suite pins the Malmuth–Harville arithmetic; what is worth asserting here
//! is the sidecar's contracts around it:
//!
//! 1. **The configuration actually reaches the solve.** The same spot solved with and without
//!    `icm` must answer in different units — otherwise everything else here is theater.
//! 2. **The $ scale is surfaced.** `icmPotValue` is what a host renders targets against, and
//!    `targetExploitability` must be computed from it, not from the chip pot.
//! 3. **Reload paths re-apply the effect.** The file format records no ICM, so the
//!    silent-corruption paths are a released job reloading its game, and `open` — with the
//!    parameter for the honest reopen, and *without* it for the documented hazard, which must
//!    be real and detectable.

use std::sync::Arc;
use std::time::{Duration, Instant};

use pkwiz_solver::spot::{BoardSpec, IcmSpec, RangeSpec, Sizing, StreetSizing};
use pkwiz_solver::{JobStatus, Jobs, Phase, Silent, Spot, Stop};

/// The 3-max situation every test here uses: two contestants and a big-stacked bystander, so
/// busts move real equity to a third party and the game is genuinely non-zero-sum in $.
fn icm_spec() -> IcmSpec {
    IcmSpec {
        payouts: vec![500.0, 300.0, 200.0],
        stacks: vec![100.0, 400.0, 500.0],
        oop_seat: 0,
        ip_seat: 1,
    }
}

/// A river spot with an all-in line (the raise cap exceeds the stack), so bust terminals
/// exist. KK loses to exactly the AA half of villain's range, so the strategy is non-trivial.
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
            target_exploitability: None,
            target_exploitability_pct: 0.5,
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
        icm: Some(icm_spec()),
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
    let dir = std::env::temp_dir().join(format!("pkwiz-icm-{}-{label}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn an_icm_solve_reports_dollar_scale_and_non_zero_sum_evs() {
    let jobs = Jobs::new(Arc::new(Silent));

    let queued = jobs.submit(river_spot(400)).unwrap();
    // The $ scale of every exploitability number, present from the queued frame on.
    let pot_value = queued
        .icm_pot_value
        .expect("an ICM job carries icmPotValue");
    assert!(pot_value > 0.0 && pot_value < 500.0, "{pot_value}");
    // The 0.5% target scales by the pot's $ value, not its 100 chips.
    let expected_target = (pot_value * 0.5 / 100.0) as f32;
    assert!(
        (queued.target_exploitability - expected_target).abs() < 1e-9,
        "{} vs {expected_target}",
        queued.target_exploitability
    );
    // startingPot stays chips; icmPotValue is the $ scale beside it.
    assert_eq!(queued.starting_pot, 100);

    let done = finish(&jobs, queued.job_id, Duration::from_secs(60));
    assert_eq!(done.phase, Phase::Done, "{:?}", done.error);
    assert_eq!(done.icm_pot_value, Some(pot_value));

    // Busts hand equity to the bystander: the contestants' $ deltas do not cancel.
    let ev = done.ev.unwrap();
    assert!(
        (ev[0] + ev[1]).abs() > 1e-4,
        "expected non-zero-sum $ EVs, got {ev:?}"
    );

    // Node EVs are absolute tournament $: OOP's whole-range average sits near its ~$262
    // baseline (stack 100 of 1100 chips plus half the pot, against a $1000 pool), nowhere
    // near a chip count.
    let root = jobs.node(done.job_id, &[]).unwrap();
    let average_ev = root.average_ev.unwrap();
    assert!(
        (240.0..280.0).contains(&average_ev),
        "expected an absolute $ EV near the baseline, got {average_ev}"
    );

    // And the same spot without `icm` answers in chips — the configuration demonstrably
    // reached the solve.
    let mut chip_spot = river_spot(400);
    chip_spot.icm = None;
    let chip_queued = jobs.submit(chip_spot).unwrap();
    assert_eq!(chip_queued.icm_pot_value, None);
    assert!((chip_queued.target_exploitability - 0.5).abs() < 1e-9);
    let chip_done = finish(&jobs, chip_queued.job_id, Duration::from_secs(60));
    let chip_root = jobs.node(chip_done.job_id, &[]).unwrap();
    let chip_average = chip_root.average_ev.unwrap();
    assert!(
        (chip_average - average_ev).abs() > 100.0,
        "chip {chip_average} vs icm {average_ev} should be in different units"
    );
}

#[test]
fn a_released_icm_job_reapplies_the_effect_on_reload() {
    let dir = temp_dir("release");
    let path = dir.join("icm.bin");

    let jobs = Jobs::new(Arc::new(Silent));
    let mut spot = river_spot(200);
    spot.save_path = Some(path.to_string_lossy().into_owned());
    let done = finish(
        &jobs,
        jobs.submit(spot).unwrap().job_id,
        Duration::from_secs(60),
    );
    assert_eq!(done.phase, Phase::Done, "{:?}", done.error);

    let before = jobs.node(done.job_id, &[]).unwrap();

    let released = jobs.release(done.job_id).unwrap();
    assert!(!released.resident);

    // The transparent reload must re-apply the ICM effect (set_icm_effect plus one
    // re-finalize), or these $ numbers would silently come back as chips.
    let after = jobs.node(done.job_id, &[]).unwrap();
    assert!(jobs.status(done.job_id).unwrap().resident);
    assert_eq!(before.ev, after.ev, "EVs changed across release/reload");
    assert_eq!(before.ev_detail, after.ev_detail);
    assert_eq!(before.strategy, after.strategy);
    assert_eq!(before.equity, after.equity);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn open_with_icm_matches_and_open_without_it_is_the_documented_hazard() {
    let dir = temp_dir("open");
    let path = dir.join("icm.bin");

    let jobs = Jobs::new(Arc::new(Silent));
    let mut spot = river_spot(200);
    spot.save_path = Some(path.to_string_lossy().into_owned());
    let done = finish(
        &jobs,
        jobs.submit(spot).unwrap().job_id,
        Duration::from_secs(60),
    );
    assert_eq!(done.phase, Phase::Done, "{:?}", done.error);
    let before = jobs.node(done.job_id, &[]).unwrap();

    // The honest reopen: same icm, same $ numbers, and the job advertises its $ scale.
    let reopened = jobs
        .open(&path.to_string_lossy(), None, None, Some(icm_spec()))
        .unwrap();
    assert_eq!(reopened.phase, Phase::Done);
    assert!(reopened.icm_pot_value.is_some());
    let honest = jobs.node(reopened.job_id, &[]).unwrap();
    assert_eq!(honest.ev, before.ev);
    assert_eq!(honest.ev_detail, before.ev_detail);
    assert_eq!(honest.strategy, before.strategy);

    // The hazard the docs warn about, asserted real: the file records nothing about ICM, so
    // opening without the parameter reads the same strategy in chip space.
    let bare = jobs
        .open(&path.to_string_lossy(), None, None, None)
        .unwrap();
    assert_eq!(bare.icm_pot_value, None);
    let chip = jobs.node(bare.job_id, &[]).unwrap();
    assert_eq!(chip.strategy, before.strategy, "the strategy is the file's");
    let chip_average = chip.average_ev.unwrap();
    let icm_average = before.average_ev.unwrap();
    assert!(
        (chip_average - icm_average).abs() > 100.0,
        "chip {chip_average} vs icm {icm_average} should differ"
    );

    // A spec that does not match the loaded tree is refused, with the wire-named message.
    let mut wrong = icm_spec();
    wrong.stacks[0] = 90.0;
    let err = jobs
        .open(&path.to_string_lossy(), None, None, Some(wrong))
        .unwrap_err();
    assert!(
        err.to_string().contains("effectiveStack"),
        "{err}: expected the units-consistency refusal"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_bad_icm_spec_fails_the_solve_command_synchronously() {
    let jobs = Jobs::new(Arc::new(Silent));
    let mut spot = river_spot(10);
    let mut spec = icm_spec();
    spec.payouts = vec![10.0, 10.0, 10.0];
    spot.icm = Some(spec);

    let err = jobs.submit(spot).unwrap_err();
    assert_eq!(err.code(), "engine", "spot errors share the engine code");
    assert!(err.to_string().contains("flat"), "{err}");
}
