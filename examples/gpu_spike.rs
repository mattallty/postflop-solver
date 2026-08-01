//! Spike benchmark for GPU offload of the bunching-effect terminal evaluation.
//!
//! Run with:
//! ```sh
//! cargo run --release --features gpu --example gpu_spike
//! ```
//!
//! Answers three questions:
//!   1. Does the GPU produce the same numbers as the CPU reference?
//!   2. How much faster is it, at realistic batch sizes?
//!   3. How much of any speedup is really just untuned CPU code? (hence the branchless
//!      CPU variant, which removes the `match` and the f64 widening that block SIMD)
//!
//! If no compatible GPU is found the benchmark still runs the CPU paths and says so, which
//! doubles as a demonstration of the intended fallback behaviour.

use postflop_solver::gpu::*;
use std::time::Instant;

/// Opponent hand count (dot product length). 1326 = C(52,2), the worst case.
const LEN: usize = 1326;
/// Player hand count (outputs per node).
const NUM_ROWS: usize = 1326;
/// Size of the synthetic combination table, chosen to match the real one (~62MB).
const ARENA_LEN: usize = 16 << 20;

/// A workgroup per output, and dispatch dimensions cap at 65535, so keep batches under that.
const BATCHES: [usize; 4] = [1, 4, 16, 48];

struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() % 10_000) as f32 / 10_000.0
    }
}

fn best_of<F: FnMut() -> Vec<f32>>(runs: usize, mut f: F) -> (f64, Vec<f32>) {
    // Warm up: the first GPU dispatch pays lazy pipeline compilation, and the first CPU pass
    // warms caches and spins up the rayon pool. Timing those would flatter later batches.
    let _ = f();
    let _ = f();

    let mut best = f64::MAX;
    let mut out = Vec::new();
    for _ in 0..runs {
        let t = Instant::now();
        let r = f();
        let e = t.elapsed().as_secs_f64();
        if e < best {
            best = e;
        }
        out = r;
    }
    (best, out)
}

fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    let mut worst = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        let scale = x.abs().max(y.abs()).max(1e-6);
        worst = worst.max(d / scale);
    }
    worst
}

fn main() {
    println!(
        "Building synthetic data ({} MB arena)...",
        (ARENA_LEN * 4) >> 20
    );
    let mut rng = Rng(0x1234_5678_9ABC_DEF0);

    let arena: Vec<f32> = (0..ARENA_LEN).map(|_| rng.next_f32()).collect();
    let max_batch = *BATCHES.iter().max().unwrap();
    let cfreach: Vec<f32> = (0..max_batch * LEN).map(|_| rng.next_f32()).collect();
    let cond: Vec<u16> = (0..max_batch * LEN)
        .map(|_| (rng.next_u32() % 8000) as u16)
        .collect();

    // Offsets into the arena, with ~5% of outputs marked "no valid combination".
    let offsets: Vec<u32> = (0..max_batch * NUM_ROWS)
        .map(|_| {
            if rng.next_u32() % 100 < 5 {
                u32::MAX
            } else {
                (rng.next_u32() as usize % (ARENA_LEN - LEN)) as u32
            }
        })
        .collect();

    let gpu = GpuEvaluator::try_new();
    match &gpu {
        Some(g) => println!("GPU detected: {}\n", g.adapter_info()),
        None => println!("No compatible GPU found — CPU paths only (fallback path exercised)\n"),
    }

    let arena_buf = gpu.as_ref().map(|g| {
        let t = Instant::now();
        let b = g.upload_arena(&arena);
        println!(
            "One-time arena upload: {:.1} ms ({} MB)\n",
            t.elapsed().as_secs_f64() * 1e3,
            (ARENA_LEN * 4) >> 20
        );
        b
    });

    println!(
        "{:>6}  {:>9}  {:>11}  {:>11}  {:>11}  {:>11}  {:>8}  {:>9}",
        "nodes",
        "outputs",
        "cpu ref",
        "cpu branchl",
        "gpu total",
        "gpu kernel",
        "gpu/ref",
        "kern/ref"
    );
    println!("{}", "-".repeat(96));

    for &nodes in &BATCHES {
        let batch = EvalBatch {
            arena: &arena,
            cfreach: &cfreach[..nodes * LEN],
            cond: &cond[..nodes * LEN],
            offsets: &offsets[..nodes * NUM_ROWS],
            len: LEN,
            num_rows: NUM_ROWS,
            threshold: 4000,
            less: -1.5,
            greater: 2.5,
            equal: 0.25,
        };

        let (t_ref, r_ref) = best_of(5, || cpu_batched(&batch));
        let (t_bl, r_bl) = best_of(5, || cpu_batched_branchless(&batch));

        let (t_gpu, r_gpu) = match (&gpu, &arena_buf) {
            (Some(g), Some(buf)) => {
                let (t, r) = best_of(5, || g.eval_batch(&batch, buf));
                (Some(t), Some(r))
            }
            _ => (None, None),
        };

        // Pure kernel time, excluding per-call buffer construction and readback.
        let t_kernel = match (&gpu, &arena_buf) {
            (Some(g), Some(buf)) => Some(g.time_compute_only(&batch, buf, 20)),
            _ => None,
        };

        let fmt = |t: Option<f64>| match t {
            Some(v) => format!("{:.2} ms", v * 1e3),
            None => "-".to_string(),
        };
        let ratio = |a: Option<f64>, b: f64| match a {
            Some(v) if v > 0.0 => format!("{:.2}x", b / v),
            _ => "-".to_string(),
        };

        println!(
            "{:>6}  {:>9}  {:>11}  {:>11}  {:>11}  {:>11}  {:>8}  {:>9}",
            nodes,
            nodes * NUM_ROWS,
            fmt(Some(t_ref)),
            fmt(Some(t_bl)),
            fmt(t_gpu),
            fmt(t_kernel),
            ratio(t_gpu, t_ref),
            ratio(t_kernel, t_ref),
        );

        // Correctness, checked against the reference on the largest batch.
        if nodes == *BATCHES.iter().max().unwrap() {
            println!();
            println!(
                "accuracy vs cpu reference:  branchless max rel err = {:.2e}",
                max_rel_err(&r_ref, &r_bl)
            );
            if let Some(r) = &r_gpu {
                println!(
                    "                            gpu        max rel err = {:.2e}",
                    max_rel_err(&r_ref, r)
                );
            }
        }
    }
}
