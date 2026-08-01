//! An example that builds and solves a game from an external JSON configuration file.
//!
//! This demonstrates how to drive `postflop-solver` from data (e.g., produced by another tool
//! or a UI) instead of hard-coding the configuration in Rust, as in the `basic` example.
//!
//! Usage:
//! ```sh
//! cargo run --release --example from_config -- [path/to/config.json]
//! ```
//! If no path is given, `examples/config.json` is used.

use postflop_solver::*;
use serde::Deserialize;
use std::env;
use std::fs;

#[derive(Deserialize)]
struct BetSizeConfig {
    bet: String,
    raise: String,
}

#[derive(Deserialize)]
struct SolverConfig {
    oop_range: String,
    ip_range: String,
    flop: String,
    turn: Option<String>,
    river: Option<String>,

    starting_pot: i32,
    effective_stack: i32,
    #[serde(default)]
    rake_rate: f64,
    #[serde(default)]
    rake_cap: f64,

    flop_bet_sizes: BetSizeConfig,
    turn_bet_sizes: BetSizeConfig,
    river_bet_sizes: BetSizeConfig,
    turn_donk_sizes: Option<String>,
    river_donk_sizes: Option<String>,

    add_allin_threshold: f64,
    force_allin_threshold: f64,
    merging_threshold: f64,

    max_num_iterations: u32,
    target_exploitability_ratio: f32,

    #[serde(default)]
    use_compression: bool,
}

fn parse_bet_sizes(config: &BetSizeConfig) -> BetSizeOptions {
    BetSizeOptions::try_from((config.bet.as_str(), config.raise.as_str()))
        .expect("failed to parse bet size options")
}

fn main() {
    let config_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/config.json".to_string());

    let config_str = fs::read_to_string(&config_path)
        .unwrap_or_else(|e| panic!("failed to read {config_path}: {e}"));
    let config: SolverConfig =
        serde_json::from_str(&config_str).expect("failed to parse configuration file");

    let flop = flop_from_str(&config.flop).expect("invalid flop");
    let turn = config
        .turn
        .as_deref()
        .map_or(NOT_DEALT, |s| card_from_str(s).expect("invalid turn"));
    let river = config
        .river
        .as_deref()
        .map_or(NOT_DEALT, |s| card_from_str(s).expect("invalid river"));

    let card_config = CardConfig {
        range: [
            config.oop_range.parse().expect("invalid oop_range"),
            config.ip_range.parse().expect("invalid ip_range"),
        ],
        flop,
        turn,
        river,
    };

    let initial_state = if river != NOT_DEALT {
        BoardState::River
    } else if turn != NOT_DEALT {
        BoardState::Turn
    } else {
        BoardState::Flop
    };

    let flop_bet_sizes = parse_bet_sizes(&config.flop_bet_sizes);
    let turn_bet_sizes = parse_bet_sizes(&config.turn_bet_sizes);
    let river_bet_sizes = parse_bet_sizes(&config.river_bet_sizes);

    let turn_donk_sizes = config
        .turn_donk_sizes
        .as_deref()
        .map(|s| DonkSizeOptions::try_from(s).expect("failed to parse turn_donk_sizes"));
    let river_donk_sizes = config
        .river_donk_sizes
        .as_deref()
        .map(|s| DonkSizeOptions::try_from(s).expect("failed to parse river_donk_sizes"));

    let tree_config = TreeConfig {
        initial_state,
        starting_pot: config.starting_pot,
        effective_stack: config.effective_stack,
        rake_rate: config.rake_rate,
        rake_cap: config.rake_cap,
        flop_bet_sizes: [flop_bet_sizes.clone(), flop_bet_sizes],
        turn_bet_sizes: [turn_bet_sizes.clone(), turn_bet_sizes],
        river_bet_sizes: [river_bet_sizes.clone(), river_bet_sizes],
        turn_donk_sizes,
        river_donk_sizes,
        add_allin_threshold: config.add_allin_threshold,
        force_allin_threshold: config.force_allin_threshold,
        merging_threshold: config.merging_threshold,
    };

    let action_tree = ActionTree::new(tree_config).expect("failed to build action tree");
    let mut game =
        PostFlopGame::with_config(card_config, action_tree).expect("failed to build game");

    let (mem_usage, mem_usage_compressed) = game.memory_usage();
    println!(
        "Memory usage without compression (32-bit float): {:.2}GB",
        mem_usage as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "Memory usage with compression (16-bit integer): {:.2}GB",
        mem_usage_compressed as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    game.allocate_memory(config.use_compression);

    let target_exploitability = config.starting_pot as f32 * config.target_exploitability_ratio;
    let exploitability = solve(
        &mut game,
        config.max_num_iterations,
        target_exploitability,
        true,
    );
    println!("Exploitability: {exploitability:.2}");

    game.cache_normalized_weights();
    let equity = game.equity(0);
    let ev = game.expected_values(0);
    let weights = game.normalized_weights(0);
    let average_equity = compute_average(&equity, weights);
    let average_ev = compute_average(&ev, weights);
    println!("OOP average equity: {:.2}%", 100.0 * average_equity);
    println!("OOP average EV: {average_ev:.2}");
}
