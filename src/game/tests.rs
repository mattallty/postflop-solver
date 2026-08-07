use super::*;
use crate::interface::{Game, GameNode};
use crate::range::*;
use crate::save_data_into_std_write;
use crate::solver::*;
use crate::utility::*;
use crate::BunchingData;
use crate::{icm_equity, icm_pot_value, load_data_from_std_read, BetSizeOptions};
use std::collections::HashSet;

#[test]
fn all_check_all_range() {
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.play(0);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.play(0);
    assert!(game.is_chance_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.play(usize::MAX);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_terminal_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);
}

#[test]
fn one_raise_all_range() {
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 37.5).abs() < 1e-4);
    assert!((ev_ip - 22.5).abs() < 1e-4);

    game.play(0);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 37.5).abs() < 1e-4);
    assert!((ev_ip - 22.5).abs() < 1e-4);

    game.play(0);
    assert!(game.is_chance_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 37.5).abs() < 1e-4);
    assert!((ev_ip - 22.5).abs() < 1e-4);

    game.play(usize::MAX);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 37.5).abs() < 1e-4);
    assert!((ev_ip - 22.5).abs() < 1e-4);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(1);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 75.0).abs() < 1e-4);
    assert!((ev_ip - 15.0).abs() < 1e-4);

    game.play(1);
    assert!(game.is_terminal_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 60.0).abs() < 1e-4);
    assert!((ev_ip - 60.0).abs() < 1e-4);
}

#[test]
fn one_raise_all_range_compressed() {
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(true);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 37.5).abs() < 1e-2);
    assert!((ev_ip - 22.5).abs() < 1e-2);

    game.play(0);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 37.5).abs() < 1e-2);
    assert!((ev_ip - 22.5).abs() < 1e-2);

    game.play(0);
    assert!(game.is_chance_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 37.5).abs() < 1e-2);
    assert!((ev_ip - 22.5).abs() < 1e-2);

    game.play(usize::MAX);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 37.5).abs() < 1e-2);
    assert!((ev_ip - 22.5).abs() < 1e-2);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(1);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 75.0).abs() < 1e-2);
    assert!((ev_ip - 15.0).abs() < 1e-2);

    game.play(1);
    assert!(game.is_terminal_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 60.0).abs() < 1e-2);
    assert!((ev_ip - 60.0).abs() < 1e-2);
}

#[test]
fn one_raise_all_range_with_turn() {
    let card_config = CardConfig {
        flop: flop_from_str("Td9d6h").unwrap(),
        range: [Range::ones(); 2],
        turn: card_from_str("Qc").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: 60,
        effective_stack: 970,
        river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_equity_oop = compute_average(&game.equity(0), weights_oop);
    let root_equity_ip = compute_average(&game.equity(1), weights_ip);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_equity_oop - 0.5).abs() < 1e-5);
    assert!((root_equity_ip - 0.5).abs() < 1e-5);
    assert!((root_ev_oop - 37.5).abs() < 1e-4);
    assert!((root_ev_ip - 22.5).abs() < 1e-4);
}

#[test]
fn one_raise_all_range_with_river() {
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: card_from_str("Qc").unwrap(),
        river: card_from_str("7s").unwrap(),
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 60,
        effective_stack: 970,
        river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 37.5).abs() < 1e-4);
    assert!((ev_ip - 22.5).abs() < 1e-4);

    game.play(0);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.play(0);
    assert!(game.is_terminal_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.back_to_root();
    game.play(1);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 75.0).abs() < 1e-4);
    assert!((ev_ip - 15.0).abs() < 1e-4);

    game.play(0);
    assert!(game.is_terminal_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!(game.is_terminal_node());
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 90.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);
}

#[test]
fn always_win() {
    // be careful for straight flushes
    let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), lose_range_str.parse().unwrap()],
        flop: flop_from_str("AcAdKh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 1.0).abs() < 1e-5);
    assert!((equity_ip - 0.0).abs() < 1e-5);
    assert!((ev_oop - 60.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_terminal_node());

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 1.0).abs() < 1e-5);
    assert!((equity_ip - 0.0).abs() < 1e-5);
    assert!((ev_oop - 60.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);
}

#[test]
fn always_win_raked() {
    // be careful for straight flushes
    let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), lose_range_str.parse().unwrap()],
        flop: flop_from_str("AcAdKh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        rake_rate: 0.05,
        rake_cap: 10.0,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((ev_oop - 57.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_terminal_node());

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((ev_oop - 57.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);
}

#[test]
fn always_lose() {
    // be careful for straight flushes
    let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
    let card_config = CardConfig {
        range: [lose_range_str.parse().unwrap(), "AA".parse().unwrap()],
        flop: flop_from_str("AcAdKh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_equity_oop = compute_average(&game.equity(0), weights_oop);
    let root_equity_ip = compute_average(&game.equity(1), weights_ip);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_equity_oop - 0.0).abs() < 1e-5);
    assert!((root_equity_ip - 1.0).abs() < 1e-5);
    assert!((root_ev_oop - 0.0).abs() < 1e-4);
    assert!((root_ev_ip - 60.0).abs() < 1e-4);
}

#[test]
fn always_lose_raked() {
    // be careful for straight flushes
    let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
    let card_config = CardConfig {
        range: [lose_range_str.parse().unwrap(), "AA".parse().unwrap()],
        flop: flop_from_str("AcAdKh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        rake_rate: 0.05,
        rake_cap: 10.0,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_ev_oop - 0.0).abs() < 1e-4);
    assert!((root_ev_ip - 57.0).abs() < 1e-4);
}

#[test]
fn always_tie() {
    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), "AA".parse().unwrap()],
        flop: flop_from_str("2c6dTh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_equity_oop = compute_average(&game.equity(0), weights_oop);
    let root_equity_ip = compute_average(&game.equity(1), weights_ip);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_equity_oop - 0.5).abs() < 1e-5);
    assert!((root_equity_ip - 0.5).abs() < 1e-5);
    assert!((root_ev_oop - 30.0).abs() < 1e-4);
    assert!((root_ev_ip - 30.0).abs() < 1e-4);
}

#[test]
fn always_tie_raked() {
    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), "AA".parse().unwrap()],
        flop: flop_from_str("2c6dTh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        rake_rate: 0.05,
        rake_cap: 10.0,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_ev_oop - 28.5).abs() < 1e-4);
    assert!((root_ev_ip - 28.5).abs() < 1e-4);
}

#[test]
fn no_assignment() {
    let card_config = CardConfig {
        range: ["TT".parse().unwrap(), "TT".parse().unwrap()],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let game = PostFlopGame::with_config(card_config, action_tree);
    assert!(game.is_err());
}

#[test]
fn remove_lines() {
    use crate::bet_size::BetSizeOptions;
    let card_config = CardConfig {
        range: ["TT+,AKo,AQs+".parse().unwrap(), "AA".parse().unwrap()],
        flop: flop_from_str("2c6dTh").unwrap(),
        ..Default::default()
    };

    // simple tree: force checks on flop, and only use 1/2 pot bets on turn and river
    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        turn_bet_sizes: [
            BetSizeOptions::try_from(("50%", "")).unwrap(),
            Default::default(),
        ],
        river_bet_sizes: [
            BetSizeOptions::try_from(("50%", "")).unwrap(),
            Default::default(),
        ],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    let lines = vec![
        vec![
            Action::Check,
            Action::Check,
            Action::Chance(2),
            Action::Check,
        ],
        vec![
            Action::Check,
            Action::Check,
            Action::Chance(2),
            Action::Bet(30),
            Action::Call,
            Action::Chance(3),
            Action::Bet(60),
        ],
    ];

    let res = game.remove_lines(&lines);
    assert!(res.is_ok());

    game.allocate_memory(false);

    // check that the turn line is removed
    game.apply_history(&[0, 0, 2]);
    assert_eq!(game.available_actions(), vec![Action::Bet(30)]);

    // check that other turn lines are correct
    game.apply_history(&[0, 0, 3]);
    assert_eq!(
        game.available_actions(),
        vec![Action::Check, Action::Bet(30)]
    );

    // check that the river line is removed
    game.apply_history(&[0, 0, 2, 0, 1, 3]);
    assert_eq!(game.available_actions(), vec![Action::Check]);

    // check that other river lines are correct
    game.apply_history(&[0, 0, 2, 0, 1, 4]);
    assert_eq!(
        game.available_actions(),
        vec![Action::Check, Action::Bet(60)]
    );

    game.apply_history(&[0, 0, 3, 1, 1, 4]);
    assert_eq!(
        game.available_actions(),
        vec![Action::Check, Action::Bet(60)]
    );

    // check that `solve()` does not crash
    solve(&mut game, 10, 0.01, false);
}

#[test]
fn isomorphism_monotone() {
    let oop_range = "88+,A8s+,A5s-A2s:0.5,AJo+,ATo:0.75,K9s+,KQo,KJo:0.75,KTo:0.25,Q9s+,QJo:0.5,J8s+,JTo:0.25,T8s+,T7s:0.45,97s+,96s:0.45,87s,86s:0.75,85s:0.45,75s+:0.75,74s:0.45,65s:0.75,64s:0.5,63s:0.45,54s:0.75,53s:0.5,52s:0.45,43s:0.5,42s:0.45,32s:0.45";
    let ip_range = "AA:0.25,99-22,AJs-A2s,AQo-A8o,K2s+,K9o+,Q2s+,Q9o+,J6s+,J9o+,T6s+,T9o,96s+,95s:0.5,98o,86s+,85s:0.5,75s+,74s:0.5,64s+,63s:0.5,54s,53s:0.5,43s";

    let card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop: flop_from_str("QhJh2h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 100,
        effective_stack: 100,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    let mut check = |history: &[usize],
                     expected_turn_swap: Option<u8>,
                     expected_river_swap: Option<(u8, u8)>| {
        game.apply_history(history);
        game.cache_normalized_weights();
        let weights = game.normalized_weights(0);
        let ev = game.expected_values(0);
        weights.iter().zip(ev.iter()).for_each(|(&w, &v)| {
            assert!(!(w > 0.0 && v == 50.0));
        });
        assert_eq!(game.turn_swap, expected_turn_swap);
        assert_eq!(game.river_swap, expected_river_swap);
    };

    check(&[0, 0, 4], None, None);
    check(&[0, 0, 5], Some(1), None);
    check(&[0, 0, 6], None, None);
    check(&[0, 0, 7], Some(3), None);

    check(&[0, 0, 4, 0, 0, 8], None, None);
    check(&[0, 0, 4, 0, 0, 9], None, None);
    check(&[0, 0, 4, 0, 0, 10], None, None);
    check(&[0, 0, 4, 0, 0, 11], None, Some((0, 3)));

    check(&[0, 0, 5, 0, 0, 8], Some(1), None);
    check(&[0, 0, 5, 0, 0, 9], Some(1), None);
    check(&[0, 0, 5, 0, 0, 10], Some(1), None);
    check(&[0, 0, 5, 0, 0, 11], Some(1), Some((1, 3)));

    check(&[0, 0, 6, 0, 0, 8], None, None);
    check(&[0, 0, 6, 0, 0, 9], None, Some((2, 1)));
    check(&[0, 0, 6, 0, 0, 10], None, None);
    check(&[0, 0, 6, 0, 0, 11], None, Some((2, 3)));

    check(&[0, 0, 7, 0, 0, 8], Some(3), Some((3, 1)));
    check(&[0, 0, 7, 0, 0, 9], Some(3), None);
    check(&[0, 0, 7, 0, 0, 10], Some(3), None);
    check(&[0, 0, 7, 0, 0, 11], Some(3), None);
}

#[test]
fn node_locking() {
    let card_config = CardConfig {
        range: ["AsAh,QsQh".parse().unwrap(), "KsKh".parse().unwrap()],
        flop: flop_from_str("2s3h4d").unwrap(),
        turn: card_from_str("6c").unwrap(),
        river: card_from_str("7c").unwrap(),
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 20,
        effective_stack: 10,
        river_bet_sizes: [("a", "").try_into().unwrap(), ("a", "").try_into().unwrap()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    game.play(1); // all-in
    game.lock_current_strategy(&[0.25, 0.75]); // 25% fold, 75% call
    game.back_to_root();

    solve(&mut game, 1000, 0.0, false);
    game.cache_normalized_weights();

    let ev_oop = game.expected_values(0);
    let ev_ip = game.expected_values(1);
    assert!((ev_oop[0] - 0.0).abs() < 1e-2);
    assert!((ev_oop[1] - 27.5).abs() < 5e-2);
    assert!((ev_ip[0] - 6.25).abs() < 1e-2);

    let strategy_oop = game.strategy();
    assert!((strategy_oop[0] - 1.0).abs() < 1e-3); // QQ check
    assert!((strategy_oop[1] - 0.0).abs() < 1e-3); // AA check
    assert!((strategy_oop[2] - 0.0).abs() < 1e-3); // QQ bet
    assert!((strategy_oop[3] - 1.0).abs() < 1e-3); // AA bet

    game.allocate_memory(false);
    game.play(1); // all-in
    game.lock_current_strategy(&[0.5, 0.5]); // 50% fold, 50% call
    game.back_to_root();

    solve(&mut game, 1000, 0.0, false);
    game.cache_normalized_weights();

    let ev_oop = game.expected_values(0);
    let ev_ip = game.expected_values(1);
    assert!((ev_oop[0] - 5.0).abs() < 1e-2);
    assert!((ev_oop[1] - 25.0).abs() < 5e-2);
    assert!((ev_ip[0] - 5.0).abs() < 1e-2);

    let strategy_oop = game.strategy();
    assert!((strategy_oop[0] - 0.0).abs() < 1e-3); // QQ check
    assert!((strategy_oop[1] - 0.0).abs() < 1e-3); // AA check
    assert!((strategy_oop[2] - 1.0).abs() < 1e-3); // QQ bet
    assert!((strategy_oop[3] - 1.0).abs() < 1e-3); // AA bet
}

#[test]
fn node_locking_partial() {
    let card_config = CardConfig {
        range: ["AsAh,QsQh,JsJh".parse().unwrap(), "KsKh".parse().unwrap()],
        flop: flop_from_str("2s3h4d").unwrap(),
        turn: card_from_str("6c").unwrap(),
        river: card_from_str("7c").unwrap(),
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 10,
        effective_stack: 10,
        river_bet_sizes: [("a", "").try_into().unwrap(), ("a", "").try_into().unwrap()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    game.lock_current_strategy(&[0.8, 0.0, 0.0, 0.2, 0.0, 0.0]); // JJ -> 80% check, 20% all-in

    solve(&mut game, 1000, 0.0, false);
    game.cache_normalized_weights();

    let ev_oop = game.expected_values(0);
    let ev_ip = game.expected_values(1);
    assert!((ev_oop[0] - 0.0).abs() < 1e-2);
    assert!((ev_oop[1] - 0.0).abs() < 1e-2);
    assert!((ev_oop[2] - 15.0).abs() < 5e-2);
    assert!((ev_ip[0] - 5.0).abs() < 1e-2);

    let strategy_oop = game.strategy();
    assert!((strategy_oop[0] - 0.8).abs() < 1e-3); // JJ check
    assert!((strategy_oop[1] - 0.7).abs() < 1e-3); // QQ check
    assert!((strategy_oop[2] - 0.0).abs() < 1e-3); // AA check
    assert!((strategy_oop[3] - 0.2).abs() < 1e-3); // JJ bet
    assert!((strategy_oop[4] - 0.3).abs() < 1e-3); // QQ bet
    assert!((strategy_oop[5] - 1.0).abs() < 1e-3); // AA bet
}

#[test]
fn node_locking_isomorphism() {
    let card_config = CardConfig {
        range: ["AKs".parse().unwrap(), "AKs".parse().unwrap()],
        flop: flop_from_str("2c3c4c").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 10,
        effective_stack: 10,
        river_bet_sizes: [("a", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    game.apply_history(&[0, 0, 15, 0, 0, 14]); // Turn: Spades, River: Hearts
    game.lock_current_strategy(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]); // AhKh -> check

    finalize(&mut game);

    game.apply_history(&[0, 0, 13, 0, 0, 14]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5]
    );

    game.apply_history(&[0, 0, 13, 0, 0, 15]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0]
    );

    game.apply_history(&[0, 0, 14, 0, 0, 13]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5]
    );

    game.apply_history(&[0, 0, 14, 0, 0, 15]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0]
    );

    game.apply_history(&[0, 0, 15, 0, 0, 13]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5]
    );

    game.apply_history(&[0, 0, 15, 0, 0, 14]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5]
    );
}

#[test]
fn set_bunching_effect() {
    let flop = flop_from_str("Td9d6h").unwrap();
    let card_config = CardConfig {
        flop,
        range: [Range::ones(); 2],
        turn: card_from_str("Qc").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: 60,
        effective_stack: 970,
        river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    let co_range = "33:0.59,22:0.635,A8o:0.265,A7o-A6o,A5o:0.445,A4o-A2o,K2s,K9o:0.905,K8o-K2o,Q4s-Q2s,Q9o-Q2o,J6s-J2s,J9o:0.88,J8o-J2o,T7s:0.405,T6s-T2s,T9o:0.96,T8o-T2o,96s-92s,92o+,86s:0.57,85s-82s,82o+,76s:0.37,75s-72s,72o+,65s:0.475,64s-62s,62o+,54s:0.68,53s-52s,52o+,42+,32";
    let sb_range = "66:0.46,55:0.821,44:0.92,33:0.93,22:0.925,A6s:0.73,A3s:0.47,A2s,ATo:0.105,A9o-A2o,K8s:0.795,K7s,K6s:0.85,K5s:0.965,K4s-K2s,KJo:0.085,KTo:0.645,K9o-K2o,Q8s-Q2s,QJo:0.765,QTo-Q2o,J8s-J2s,J2o+,T8s:0.69,T7s-T2s,T2o+,98s:0.905,97s-92s,92o+,87s:0.78,86s-82s,82o+,76s:0.77,75s-72s,72o+,65s:0.845,64s-62s,62o+,54s:0.735,53s-52s,52o+,42+,32";

    let mut bunching_data = BunchingData::new(
        &[co_range.parse().unwrap(), sb_range.parse().unwrap()],
        flop,
    )
    .unwrap();

    bunching_data.process(false);
    game.set_bunching_effect(&bunching_data).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    let current_ev = compute_current_ev(&game);
    assert!((current_ev[0] - 7.5).abs() < 1e-4);
    assert!((current_ev[1] - -7.5).abs() < 1e-4);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_equity_oop = compute_average(&game.equity(0), weights_oop);
    let root_equity_ip = compute_average(&game.equity(1), weights_ip);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_equity_oop - 0.5).abs() < 1e-5);
    assert!((root_equity_ip - 0.5).abs() < 1e-5);
    assert!((root_ev_oop - 37.5).abs() < 1e-4);
    assert!((root_ev_ip - 22.5).abs() < 1e-4);
}

#[test]
fn set_bunching_effect_always_win() {
    let flop = flop_from_str("AcAdKh").unwrap();
    let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";

    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), lose_range_str.parse().unwrap()],
        flop,
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    let co_range = "33:0.59,22:0.635,A8o:0.265,A7o-A6o,A5o:0.445,A4o-A2o,K2s,K9o:0.905,K8o-K2o,Q4s-Q2s,Q9o-Q2o,J6s-J2s,J9o:0.88,J8o-J2o,T7s:0.405,T6s-T2s,T9o:0.96,T8o-T2o,96s-92s,92o+,86s:0.57,85s-82s,82o+,76s:0.37,75s-72s,72o+,65s:0.475,64s-62s,62o+,54s:0.68,53s-52s,52o+,42+,32";
    let sb_range = "66:0.46,55:0.821,44:0.92,33:0.93,22:0.925,A6s:0.73,A3s:0.47,A2s,ATo:0.105,A9o-A2o,K8s:0.795,K7s,K6s:0.85,K5s:0.965,K4s-K2s,KJo:0.085,KTo:0.645,K9o-K2o,Q8s-Q2s,QJo:0.765,QTo-Q2o,J8s-J2s,J2o+,T8s:0.69,T7s-T2s,T2o+,98s:0.905,97s-92s,92o+,87s:0.78,86s-82s,82o+,76s:0.77,75s-72s,72o+,65s:0.845,64s-62s,62o+,54s:0.735,53s-52s,52o+,42+,32";

    let mut bunching_data = BunchingData::new(
        &[co_range.parse().unwrap(), sb_range.parse().unwrap()],
        flop,
    )
    .unwrap();

    bunching_data.process(false);
    game.set_bunching_effect(&bunching_data).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    let current_ev = compute_current_ev(&game);
    assert!((current_ev[0] - 30.0).abs() < 1e-4);
    assert!((current_ev[1] - -30.0).abs() < 1e-4);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 1.0).abs() < 1e-5);
    assert!((equity_ip - 0.0).abs() < 1e-5);
    assert!((ev_oop - 60.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_terminal_node());

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 1.0).abs() < 1e-5);
    assert!((equity_ip - 0.0).abs() < 1e-5);
    assert!((ev_oop - 60.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);
}

#[test]
#[ignore]
fn solve_pio_preset_normal() {
    let oop_range = "88+,A8s+,A5s-A2s:0.5,AJo+,ATo:0.75,K9s+,KQo,KJo:0.75,KTo:0.25,Q9s+,QJo:0.5,J8s+,JTo:0.25,T8s+,T7s:0.45,97s+,96s:0.45,87s,86s:0.75,85s:0.45,75s+:0.75,74s:0.45,65s:0.75,64s:0.5,63s:0.45,54s:0.75,53s:0.5,52s:0.45,43s:0.5,42s:0.45,32s:0.45";
    let ip_range = "AA:0.25,99-22,AJs-A2s,AQo-A8o,K2s+,K9o+,Q2s+,Q9o+,J6s+,J9o+,T6s+,T9o,96s+,95s:0.5,98o,86s+,85s:0.5,75s+,74s:0.5,64s+,63s:0.5,54s,53s:0.5,43s";

    let card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop: flop_from_str("QsJh2h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 180,
        effective_stack: 910,
        flop_bet_sizes: [
            ("52%", "45%").try_into().unwrap(),
            ("52%", "45%").try_into().unwrap(),
        ],
        turn_bet_sizes: [
            ("55%", "45%").try_into().unwrap(),
            ("55%", "45%").try_into().unwrap(),
        ],
        river_bet_sizes: [
            ("70%", "45%").try_into().unwrap(),
            ("70%", "45%").try_into().unwrap(),
        ],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    println!(
        "memory usage: {:.2}GB",
        game.memory_usage().0 as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    game.allocate_memory(false);

    solve(&mut game, 1000, 180.0 * 0.001, true);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_equity_oop = compute_average(&game.equity(0), weights_oop);
    let root_equity_ip = compute_average(&game.equity(1), weights_ip);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    // verified by PioSOLVER Free
    assert!((root_equity_oop - 0.55347).abs() < 1e-5);
    assert!((root_equity_ip - 0.44653).abs() < 1e-5);
    assert!((root_ev_oop - 105.11).abs() < 0.2);
    assert!((root_ev_ip - 74.89).abs() < 0.2);
}

#[test]
#[ignore]
fn solve_pio_preset_raked() {
    let oop_range = "88+,A8s+,A5s-A2s:0.5,AJo+,ATo:0.75,K9s+,KQo,KJo:0.75,KTo:0.25,Q9s+,QJo:0.5,J8s+,JTo:0.25,T8s+,T7s:0.45,97s+,96s:0.45,87s,86s:0.75,85s:0.45,75s+:0.75,74s:0.45,65s:0.75,64s:0.5,63s:0.45,54s:0.75,53s:0.5,52s:0.45,43s:0.5,42s:0.45,32s:0.45";
    let ip_range = "AA:0.25,99-22,AJs-A2s,AQo-A8o,K2s+,K9o+,Q2s+,Q9o+,J6s+,J9o+,T6s+,T9o,96s+,95s:0.5,98o,86s+,85s:0.5,75s+,74s:0.5,64s+,63s:0.5,54s,53s:0.5,43s";

    let card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop: flop_from_str("QsJh2h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 180,
        effective_stack: 910,
        rake_rate: 0.05,
        rake_cap: 30.0,
        flop_bet_sizes: [
            ("52%", "45%").try_into().unwrap(),
            ("52%", "45%").try_into().unwrap(),
        ],
        turn_bet_sizes: [
            ("55%", "45%").try_into().unwrap(),
            ("55%", "45%").try_into().unwrap(),
        ],
        river_bet_sizes: [
            ("70%", "45%").try_into().unwrap(),
            ("70%", "45%").try_into().unwrap(),
        ],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    println!(
        "memory usage: {:.2}GB",
        game.memory_usage().0 as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    game.allocate_memory(false);

    solve(&mut game, 1000, 180.0 * 0.001, true);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    // verified by PioSOLVER Free (but not theoretically guaranteed to be the same)
    assert!((root_ev_oop - 95.57).abs() < 0.2);
    assert!((root_ev_ip - 66.98).abs() < 0.2);
}

#[test]
fn visit_all_nodes() {
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    // Move to a non-root node beforehand to check that `visit` restores the current position.
    game.play(0);
    let history_before = game.history().to_vec();

    let mut num_nodes = 0;
    let mut num_terminal = 0;
    let mut num_chance = 0;
    let mut num_player = 0;
    let mut histories = HashSet::new();

    game.visit(|g| {
        num_nodes += 1;
        histories.insert(g.history().to_vec());
        if g.is_terminal_node() {
            num_terminal += 1;
        } else if g.is_chance_node() {
            num_chance += 1;
        } else {
            num_player += 1;
            assert!(!g.available_actions().is_empty());
        }
    });

    // `visit` must not change the current node.
    assert_eq!(game.history(), history_before.as_slice());

    // every node must be visited exactly once
    assert_eq!(num_nodes, histories.len());

    // every visited node must be a descendant of the starting node
    assert!(histories.iter().all(|h| h.starts_with(&history_before)));

    // sanity checks: the subtree rooted at the current node (after the first "check") should
    // contain a mix of player, chance, and terminal nodes.
    assert_eq!(num_nodes, num_terminal + num_chance + num_player);
    assert!(num_terminal > 0);
    assert!(num_chance > 0);
    assert!(num_player > 0);
}

/// Builds a deliberately tiny river game whose whole tree can be enumerated by hand:
///
/// ```text
/// []      OOP: Check | Bet(30)
/// [0]      |- OOP checks -> IP: Check | Bet(30)
/// [0,0]    |   |- IP checks -> showdown
/// [0,1]    |   `- IP bets   -> OOP: Fold | Call
/// [0,1,0]  |       |- OOP folds
/// [0,1,1]  |       `- OOP calls
/// [1]     `- OOP bets -> IP: Fold | Call
/// [1,0]        |- IP folds
/// [1,1]        `- IP calls
/// ```
///
/// i.e. 9 nodes: 4 player nodes, 5 terminal nodes, and no chance node.
fn tiny_river_game() -> PostFlopGame {
    let card_config = CardConfig {
        range: [
            "AA,KK".parse::<Range>().unwrap(),
            "QQ,JJ".parse::<Range>().unwrap(),
        ],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: card_from_str("As").unwrap(),
        river: card_from_str("2c").unwrap(),
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 60,
        effective_stack: 300,
        // a single bet size and no raise size, so the tree stays hand-enumerable
        river_bet_sizes: [
            BetSizeOptions::try_from(("50%", "")).unwrap(),
            BetSizeOptions::try_from(("50%", "")).unwrap(),
        ],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);
    game
}

#[test]
fn visit_exact_node_set() {
    let mut game = tiny_river_game();

    let mut num_nodes = 0;
    let mut num_terminal = 0;
    let mut num_player = 0;
    let mut histories = HashSet::new();

    game.visit(|g| {
        num_nodes += 1;
        histories.insert(g.history().to_vec());
        if g.is_terminal_node() {
            num_terminal += 1;
        } else {
            assert!(!g.is_chance_node());
            num_player += 1;
        }
    });

    let expected: HashSet<Vec<usize>> = [
        vec![],
        vec![0],
        vec![0, 0],
        vec![0, 1],
        vec![0, 1, 0],
        vec![0, 1, 1],
        vec![1],
        vec![1, 0],
        vec![1, 1],
    ]
    .into_iter()
    .collect();

    // the exact node set, so neither a missed node nor a node visited twice can pass
    assert_eq!(histories, expected);
    assert_eq!(num_nodes, 9);
    assert_eq!(num_player, 4);
    assert_eq!(num_terminal, 5);
}

#[test]
fn visit_with_navigating_visitor() {
    let mut game = tiny_river_game();

    let mut baseline = Vec::new();
    game.visit(|g| baseline.push(g.history().to_vec()));

    // a visitor that navigates the tree must not affect which nodes are visited
    let mut visited = Vec::new();
    game.visit(|g| {
        visited.push(g.history().to_vec());
        g.back_to_root();
        if !g.is_terminal_node() {
            g.play(0);
        }
    });

    assert_eq!(visited, baseline);
    assert_eq!(visited.len(), 9);
    assert!(game.history().is_empty());
}

#[test]
fn visit_reads_expected_values() {
    let mut game = tiny_river_game();

    // `expected_values` is usable from within a visitor as long as the visitor caches the
    // normalized weights itself, since navigating to a node invalidates them
    let mut num_evs = 0;
    game.visit(|g| {
        if !g.is_terminal_node() && !g.is_chance_node() {
            g.cache_normalized_weights();
            let player = g.current_player();
            assert_eq!(
                g.expected_values(player).len(),
                g.private_cards(player).len()
            );
            assert_eq!(g.equity(player).len(), g.private_cards(player).len());
            num_evs += 1;
        }
    });

    assert_eq!(num_evs, 4);
}

#[test]
fn visit_with_reduced_storage() {
    let card_config = CardConfig {
        range: [
            "AA,KK".parse::<Range>().unwrap(),
            "QQ,JJ".parse::<Range>().unwrap(),
        ],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: card_from_str("As").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: 60,
        effective_stack: 60,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    // full-storage traversal descends past the river deal
    let mut full_chance = 0;
    let mut full_nodes = 0;
    game.visit(|g| {
        full_nodes += 1;
        if g.is_chance_node() {
            full_chance += 1;
        }
    });
    assert!(full_chance > 0);

    // discard the storage after the river deal
    let mut buf = Vec::new();
    game.set_target_storage_mode(BoardState::Turn).unwrap();
    save_data_into_std_write(&game, "", &mut buf, None).unwrap();
    let (mut truncated, _): (PostFlopGame, String) =
        load_data_from_std_read(&mut buf.as_slice(), None).unwrap();

    // the traversal must stop at the storage boundary rather than panic
    let mut truncated_chance = 0;
    let mut truncated_nodes = 0;
    let mut chance_histories = Vec::new();
    let mut all_histories = Vec::new();
    truncated.visit(|g| {
        truncated_nodes += 1;
        all_histories.push(g.history().to_vec());
        if g.is_chance_node() {
            truncated_chance += 1;
            chance_histories.push(g.history().to_vec());
        }
    });

    // the chance nodes at the boundary are still visited, but nothing below them is
    assert_eq!(truncated_chance, full_chance);
    assert!(truncated_nodes < full_nodes);
    assert!(!chance_histories.is_empty());

    for chance in &chance_histories {
        assert!(
            !all_histories
                .iter()
                .any(|h| h.len() > chance.len() && h.starts_with(chance)),
            "descended past the storage boundary at {chance:?}"
        );
    }
}

#[test]
fn free_memory() {
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    // freeing memory before it is allocated is a no-op
    game.free_memory();
    assert_eq!(game.is_memory_allocated(), None);

    game.allocate_memory(false);
    finalize(&mut game);

    assert_eq!(game.is_memory_allocated(), Some(false));
    assert!(game.is_solved());
    let (uncompressed, _) = game.memory_usage();
    assert!(uncompressed > 0);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let ev_oop_before = compute_average(&game.expected_values(0), weights_oop);

    // move off the root so that the cursor reset below is observable
    game.play(0);
    assert!(!game.history().is_empty());

    game.free_memory();

    // the current node must be reset to the root: the position (and the weights and caches
    // that came with it) was derived from the storage that no longer exists
    assert!(game.history().is_empty());

    // the tree/config metadata must be preserved
    assert_eq!(game.card_config().flop, flop_from_str("Td9d6h").unwrap());
    assert_eq!(game.tree_config().starting_pot, 60);
    assert_eq!(game.memory_usage(), (uncompressed, game.memory_usage().1));

    // the storage and solved status must be reset
    assert_eq!(game.is_memory_allocated(), None);
    assert!(!game.is_solved());

    // freeing again is a no-op
    game.free_memory();
    assert_eq!(game.is_memory_allocated(), None);

    // saving must fail gracefully once the memory is freed
    let mut buf = Vec::new();
    assert!(save_data_into_std_write(&game, "", &mut buf, None).is_err());

    // the game can be fully reallocated and re-solved afterward
    game.allocate_memory(false);
    finalize(&mut game);
    assert!(game.is_solved());

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let ev_oop_after = compute_average(&game.expected_values(0), weights_oop);
    assert!((ev_oop_after - ev_oop_before).abs() < 1e-4);
}

#[test]
fn visit_below_isomorphism_eliminated_turn_card() {
    // A monotone flop makes the three non-flop suits isomorphic, so most turn cards are
    // eliminated from `available_actions` in favor of a representative of another suit.
    // Entering the subtree below such a card sets the internal suit swap, which the traversal
    // must invert when it replays the chance actions it reads from the storage.
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("AsKsQs").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    // check/check to the turn chance node
    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    let chance_history = game.history().to_vec();

    let listed: HashSet<Card> = game
        .available_actions()
        .iter()
        .map(|action| match action {
            Action::Chance(card) => *card,
            _ => unreachable!(),
        })
        .collect();

    // a dealable turn card that isomorphism eliminated, and its listed same-rank representative
    let possible = game.possible_cards();
    let eliminated = (0..52)
        .map(|card| card as Card)
        .find(|card| possible & (1 << card) != 0 && !listed.contains(card))
        .unwrap();
    let representative = *listed
        .iter()
        .find(|&&card| card >> 2 == eliminated >> 2 && card != eliminated && card & 3 != 3)
        .unwrap();

    // `play` accepts the eliminated card and swaps suits internally; the traversal below it
    // must stay in actual-card coordinates
    game.play(eliminated as usize);
    let mut num_nodes = 0;
    game.visit(|g| {
        num_nodes += 1;
        let board = g.current_board();
        assert_eq!(board[3], eliminated);
        // each dealt river must be distinct from the actual board
        let mut sorted = board.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), board.len());
    });

    // isomorphic subtrees must have the same shape
    game.apply_history(&chance_history);
    game.play(representative as usize);
    let mut num_nodes_representative = 0;
    game.visit(|_| num_nodes_representative += 1);

    assert!(num_nodes > 1);
    assert_eq!(num_nodes, num_nodes_representative);
}

/// Builds a solved turn game, saves it with `BoardState::Turn` storage, and loads it back,
/// yielding a game whose node arena is truncated at the river deal.
fn truncated_turn_game() -> PostFlopGame {
    let card_config = CardConfig {
        range: [
            "AA,KK".parse::<Range>().unwrap(),
            "QQ,JJ".parse::<Range>().unwrap(),
        ],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: card_from_str("As").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: 60,
        effective_stack: 60,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    let mut buf = Vec::new();
    game.set_target_storage_mode(BoardState::Turn).unwrap();
    save_data_into_std_write(&game, "", &mut buf, None).unwrap();
    let (truncated, _): (PostFlopGame, String) =
        load_data_from_std_read(&mut buf.as_slice(), None).unwrap();
    truncated
}

#[test]
#[should_panic(expected = "Storage mode is not compatible")]
fn available_actions_panics_at_truncated_storage_boundary() {
    let mut game = truncated_turn_game();

    // check/check to the chance node that would deal the river
    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());

    // its children are the river nodes the save dropped, so reading them would run past the
    // end of the node arena
    game.available_actions();
}

#[test]
#[should_panic(expected = "partially loaded")]
fn allocate_memory_rejects_partially_loaded_game() {
    let mut game = truncated_turn_game();

    // allocating over the truncated arena would claim the missing streets are present and let
    // the solver traverse out of its bounds
    game.allocate_memory(false);
}

#[test]
fn node_storage_accessors_tolerate_unallocated_memory() {
    let card_config = CardConfig {
        range: [
            "AA,KK".parse::<Range>().unwrap(),
            "QQ,JJ".parse::<Range>().unwrap(),
        ],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: card_from_str("As").unwrap(),
        river: card_from_str("2c").unwrap(),
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 60,
        effective_stack: 300,
        river_bet_sizes: [
            BetSizeOptions::try_from(("50%", "")).unwrap(),
            BetSizeOptions::try_from(("50%", "")).unwrap(),
        ],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    // before `allocate_memory`, the storage pointers are unassigned: the safe accessors must
    // return empty slices rather than fabricate ones no allocation backs
    assert!(game.root().strategy().is_empty());
    assert!(game.root().regrets().is_empty());
    assert!(game.root().cfvalues().is_empty());
    assert!(game.root().strategy_compressed().is_empty());
    assert!(game.root().regrets_compressed().is_empty());

    game.allocate_memory(false);
    finalize(&mut game);
    assert!(!game.root().strategy().is_empty());

    // after `free_memory`, the pointers are nulled again along with the freed storage
    game.free_memory();
    assert!(game.root().strategy().is_empty());
    assert!(game.root().regrets().is_empty());
    assert!(game.root().cfvalues().is_empty());
}

#[test]
fn loading_a_tampered_bunching_file_is_an_error_not_a_panic() {
    // A freshly constructed BunchingData has phase 0 and every table empty, and its encoding
    // ends with [phase, progress, thirteen empty-vec length bytes]. Forging phase 3 at 100%
    // produces a file that claims `is_ready()` with no result tables behind the claim — which
    // used to load successfully and then panic inside `set_bunching_effect`.
    let range = "22+,A2s+,A2o+".parse::<Range>().unwrap();
    let data = BunchingData::new(&[range], flop_from_str("Td9d6h").unwrap()).unwrap();

    // `save_data_into_std_write` refuses data that is not ready, so build the file by hand:
    // the header (magic, version, no compression, bunching data type, memory estimate, memo)
    // followed by the encoded body — exactly what the writer produces for ready data.
    let config = bincode::config::standard();
    let mut buf = Vec::new();
    bincode::encode_into_std_write(0x09f1_5790u32, &mut buf, config).unwrap();
    bincode::encode_into_std_write(1u8, &mut buf, config).unwrap();
    bincode::encode_into_std_write(0u8, &mut buf, config).unwrap();
    bincode::encode_into_std_write(1u8, &mut buf, config).unwrap();
    bincode::encode_into_std_write(0u64, &mut buf, config).unwrap();
    bincode::encode_into_std_write("", &mut buf, config).unwrap();
    bincode::encode_into_std_write(&data, &mut buf, config).unwrap();

    // A fresh instance encodes phase 0 at 0% followed by thirteen empty tables — sanity-check
    // that layout, then forge "phase 3, 100%".
    let len = buf.len();
    assert_eq!(&buf[len - 15..], [0; 15]);
    buf[len - 15] = 3; // phase
    buf[len - 14] = 100; // progress_percent

    let result: Result<(BunchingData, String), String> =
        load_data_from_std_read(&mut buf.as_slice(), None);
    let err = result.err().expect("a forged ready-claim must be refused");
    assert!(err.contains("ready"), "{err}");
}

/// Flips one low and one high bit at every `step`-th byte of `original` and asserts the decode
/// answers each with `Err` or a game — never a panic.
fn assert_bit_flips_never_panic(original: &[u8], step: usize, label: &str) {
    for index in (0..original.len()).step_by(step) {
        for bit in [0x01u8, 0x80u8] {
            let mut tampered = original.to_vec();
            tampered[index] ^= bit;

            let outcome = std::panic::catch_unwind(|| {
                let result: Result<(PostFlopGame, String), String> =
                    load_data_from_std_read(&mut tampered.as_slice(), None);
                result.is_ok()
            });
            assert!(
                outcome.is_ok(),
                "{label}: flipping bit {bit:#04x} of byte {index} panicked the decoder"
            );
        }
    }
}

#[test]
fn loading_a_bit_flipped_game_file_never_panics() {
    // A file is a trust boundary: whatever a flipped bit does to the decode, the answer must be
    // `Err` or a well-formed game — never a panic, and never (checked by the validation layer
    // in `serialization.rs`) a pointer built from unchecked file contents. This exhaustive pass
    // found four distinct crashes when it was written: forged element counts, a node that was
    // its own child (unbounded recursion), impossible board cards on a node (empty
    // strength-table entries), and a nesting depth that overflowed the decode stack.
    let mut game = tiny_river_game();
    game.play(0);

    let mut original = Vec::new();
    save_data_into_std_write(&game, "memo", &mut original, None).unwrap();

    let step = (original.len() / 8192).max(1);
    assert_bit_flips_never_panic(&original, step, "river");
}

#[test]
fn loading_a_bit_flipped_turn_game_file_never_panics() {
    // The same trust-boundary sweep over a game with chance nodes, saved both in full and with
    // the river street truncated away — the truncated file exercises the storage-boundary
    // decode paths (prefix arena, prefix buffers) the river file cannot.
    let card_config = CardConfig {
        range: [
            "AA,KK".parse::<Range>().unwrap(),
            "QQ,JJ".parse::<Range>().unwrap(),
        ],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: card_from_str("As").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: 60,
        effective_stack: 60,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game.allocate_memory(false);
    finalize(&mut game);

    let mut full = Vec::new();
    save_data_into_std_write(&game, "", &mut full, None).unwrap();

    game.set_target_storage_mode(BoardState::Turn).unwrap();
    let mut truncated = Vec::new();
    save_data_into_std_write(&game, "", &mut truncated, None).unwrap();

    // Sparser than the river sweep — two files, and the ranges that dominate the bytes are
    // already swept exhaustively there.
    assert_bit_flips_never_panic(&full, (full.len() / 4096).max(2), "full turn");
    assert_bit_flips_never_panic(
        &truncated,
        (truncated.len() / 4096).max(2),
        "truncated turn",
    );
}

/// The `tiny_river_game` spot, actually solved rather than finalized at the uniform strategy.
fn solved_tiny_river_game() -> PostFlopGame {
    let card_config = CardConfig {
        range: [
            "AA,KK".parse::<Range>().unwrap(),
            "QQ,JJ".parse::<Range>().unwrap(),
        ],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: card_from_str("As").unwrap(),
        river: card_from_str("2c").unwrap(),
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 60,
        effective_stack: 300,
        river_bet_sizes: [
            BetSizeOptions::try_from(("50%", "")).unwrap(),
            BetSizeOptions::try_from(("50%", "")).unwrap(),
        ],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    solve(&mut game, 1000, 0.06, false);
    game
}

#[test]
fn best_response_total_ev_matches_compute_mes_ev() {
    let game = solved_tiny_river_game();
    let mes_ev = compute_mes_ev(&game);
    let current_ev = compute_current_ev(&game);

    for (player, &mes_ev) in mes_ev.iter().enumerate() {
        let br = game.compute_best_response(player, false);
        // the same recursion over the same inputs
        assert!(
            (br.total_ev() - mes_ev).abs() < 1e-4,
            "player {player}: {} vs {mes_ev}",
            br.total_ev(),
        );
        // the maximally exploitative strategy dominates the current one
        assert!(br.total_ev() >= current_ev[player] - 1e-4);
        assert!(!br.has_detail());
        assert!(br.memory_usage() > 0);
    }
}

#[test]
fn best_response_strategy_is_pure_and_achieves_its_value() {
    let mut game = solved_tiny_river_game();
    let br = game.compute_best_response(0, true);
    assert!(br.has_detail());

    game.cache_normalized_weights();
    let weights = game.normalized_weights(0).to_vec();
    let ev = game.br_expected_values(&br);

    // the root average recovers the whole-game best-response EV (plus the pot/2 bias)
    let average = compute_average(&ev, &weights);
    assert!(
        (average - (br.total_ev() + 30.0)).abs() < 1e-2,
        "{average} vs {}",
        br.total_ev() + 30.0
    );

    let num_hands = game.num_private_hands(0);
    let num_actions = game.available_actions().len();
    assert_eq!(num_actions, 2); // Check | Bet(30)

    let strategy = game.br_strategy(&br);
    let actions = game.br_actions(&br);
    let detail = game.br_expected_values_detail(&br);

    for hand in 0..num_hands {
        // no locks, so every hand is a pure argmax
        let action = actions[hand].unwrap();
        for a in 0..num_actions {
            let expected = if a == action { 1.0 } else { 0.0 };
            assert_eq!(strategy[a * num_hands + hand], expected);
        }

        // the claimed value is the maximum of the per-action values (the root offers no fold,
        // so the display convention zeroes no row)
        if weights[hand] > 0.0 {
            let best = (0..num_actions)
                .map(|a| detail[a * num_hands + hand])
                .fold(f32::MIN, f32::max);
            assert!((ev[hand] - best).abs() < 1e-3, "{} vs {best}", ev[hand]);
            assert!(detail[action * num_hands + hand] >= best - 1e-3);
        }
    }
}

/// Builds the `node_locking` spot with the given fold/call frequencies locked at the node
/// facing the all-in, and solves it.
fn locked_all_in_game(fold: f32, call: f32) -> PostFlopGame {
    let card_config = CardConfig {
        range: ["AsAh,QsQh".parse().unwrap(), "KsKh".parse().unwrap()],
        flop: flop_from_str("2s3h4d").unwrap(),
        turn: card_from_str("6c").unwrap(),
        river: card_from_str("7c").unwrap(),
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 20,
        effective_stack: 10,
        river_bet_sizes: [("a", "").try_into().unwrap(), ("a", "").try_into().unwrap()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    game.play(1); // all-in
    game.lock_current_strategy(&[fold, call]);
    game.back_to_root();

    solve(&mut game, 1000, 0.0, false);
    game
}

#[test]
fn best_response_respects_locked_strategies() {
    let mut game = locked_all_in_game(0.25, 0.75);

    // the best response of the locked player keeps the pinned distribution at the locked node
    let br = game.compute_best_response(1, true);
    game.play(1);
    game.cache_normalized_weights();
    assert_eq!(game.current_player(), 1);

    assert_eq!(game.br_actions(&br), vec![None]);
    let strategy = game.br_strategy(&br);
    assert!((strategy[0] - 0.25).abs() < 1e-6);
    assert!((strategy[1] - 0.75).abs() < 1e-6);

    // its value there is the pinned-mix value, i.e., the current strategy's own EV
    let br_ev = game.br_expected_values(&br);
    let ev = game.expected_values(1);
    assert!((br_ev[0] - ev[0]).abs() < 1e-2, "{br_ev:?} vs {ev:?}");

    // the total EV still matches `compute_mes_ev`, which honours locks the same way
    assert!((br.total_ev() - compute_mes_ev(&game)[1]).abs() < 1e-4);
}

#[test]
fn best_response_of_the_opponent_tracks_the_locked_strategy() {
    // folding only 25% makes the bluff -EV, so QQ's best response is Check while AA's is the
    // all-in (hand order: QQ, AA)
    let game = locked_all_in_game(0.25, 0.75);
    let br = game.compute_best_response(0, true);
    assert_eq!(game.br_actions(&br), vec![Some(0), Some(1)]);

    // folding half makes the bluff +EV, so QQ's best response flips to the all-in
    let game = locked_all_in_game(0.5, 0.5);
    let br = game.compute_best_response(0, true);
    assert_eq!(game.br_actions(&br), vec![Some(1), Some(1)]);
}

#[test]
fn best_response_breaks_ties_to_the_lowest_action_index() {
    // OOP holds the losing range and IP is locked to fold against either bet size: both bets
    // then win exactly the starting pot, so their values tie bit-for-bit, and checking loses
    // the showdown. Every tie must resolve to the lower action index.
    let card_config = CardConfig {
        range: [
            "QQ,JJ".parse::<Range>().unwrap(),
            "AA,KK".parse::<Range>().unwrap(),
        ],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: card_from_str("As").unwrap(),
        river: card_from_str("2c").unwrap(),
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 60,
        effective_stack: 300,
        river_bet_sizes: [
            BetSizeOptions::try_from(("50%,100%", "")).unwrap(),
            BetSizeOptions::try_from(("50%", "")).unwrap(),
        ],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    assert_eq!(
        game.available_actions(),
        vec![Action::Check, Action::Bet(30), Action::Bet(60)]
    );

    for action in [1, 2] {
        game.apply_history(&[action]);
        let num_hands = game.num_private_hands(1);
        let mut lock = vec![0.0; 2 * num_hands];
        lock[..num_hands].fill(1.0); // 100% fold
        game.lock_current_strategy(&lock);
    }
    game.back_to_root();
    finalize(&mut game);

    let br = game.compute_best_response(0, true);
    game.cache_normalized_weights();

    let num_hands = game.num_private_hands(0);
    let detail = game.br_expected_values_detail(&br);
    assert_eq!(
        detail[num_hands..2 * num_hands],
        detail[2 * num_hands..3 * num_hands],
        "the two bet sizes win the same pot against a locked fold"
    );
    assert_eq!(game.br_actions(&br), vec![Some(1); num_hands]);
}

#[test]
fn best_response_below_isomorphism_eliminated_cards() {
    // the `node_locking_isomorphism` spot: a monotone flop makes the three non-club suits
    // isomorphic, so most turn/river deals are answered through a suit swap
    let card_config = CardConfig {
        range: ["AKs".parse().unwrap(), "AKs".parse().unwrap()],
        flop: flop_from_str("2c3c4c").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 10,
        effective_stack: 10,
        river_bet_sizes: [("a", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    game.apply_history(&[0, 0, 15, 0, 0, 14]); // Turn: 5s, River: 5h
    game.lock_current_strategy(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]); // AhKh -> check

    finalize(&mut game);

    let br = game.compute_best_response(0, true);

    // where the locked hand sits in view coordinates on each isomorphic line — the positions
    // `node_locking_isomorphism` pins for `strategy()`
    let cases: [(&[usize], usize); 6] = [
        (&[0, 0, 13, 0, 0, 14], 2),
        (&[0, 0, 13, 0, 0, 15], 3),
        (&[0, 0, 14, 0, 0, 13], 1),
        (&[0, 0, 14, 0, 0, 15], 3),
        (&[0, 0, 15, 0, 0, 13], 1),
        (&[0, 0, 15, 0, 0, 14], 2),
    ];

    let mut sorted_evs: Vec<Vec<f32>> = Vec::new();
    for (history, locked_index) in cases {
        game.apply_history(history);
        game.cache_normalized_weights();

        let num_hands = game.num_private_hands(0);
        let strategy = game.br_strategy(&br);
        let actions = game.br_actions(&br);

        for hand in 0..num_hands {
            if hand == locked_index {
                // the locked hand keeps its pinned pure check, and moves with the suit swap
                assert_eq!(actions[hand], None);
                assert_eq!(strategy[hand], 1.0);
                assert_eq!(strategy[num_hands + hand], 0.0);
            } else {
                // unlocked hands are one-hot at their argmax
                let action = actions[hand].unwrap();
                for a in 0..2 {
                    let expected = if a == action { 1.0 } else { 0.0 };
                    assert_eq!(strategy[a * num_hands + hand], expected);
                }
            }
        }

        // hand-wise, the best response dominates the current strategy...
        let br_ev = game.br_expected_values(&br);
        let ev = game.expected_values(0);
        for hand in 0..num_hands {
            assert!(
                br_ev[hand] >= ev[hand] - 1e-3,
                "hand {hand} at {history:?}: {} < {}",
                br_ev[hand],
                ev[hand]
            );
        }

        // ...and isomorphic lines answer the same multiset of hand values
        let mut sorted = br_ev;
        sorted.sort_by(f32::total_cmp);
        sorted_evs.push(sorted);
    }

    for pair in sorted_evs.windows(2) {
        for (a, b) in pair[0].iter().zip(pair[1].iter()) {
            assert!((a - b).abs() < 1e-5, "{:?} vs {:?}", pair[0], pair[1]);
        }
    }
}

#[test]
fn best_response_with_bunching_matches_compute_mes_ev() {
    let flop = flop_from_str("Td9d6h").unwrap();
    let card_config = CardConfig {
        range: [
            "AA,KK".parse::<Range>().unwrap(),
            "QQ,JJ".parse::<Range>().unwrap(),
        ],
        flop,
        turn: card_from_str("Qc").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: 60,
        effective_stack: 970,
        river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    let mut bunching_data = BunchingData::new(&["22+,A2s+,A2o+".parse().unwrap()], flop).unwrap();
    bunching_data.process(false);
    game.set_bunching_effect(&bunching_data).unwrap();

    game.allocate_memory(false);
    solve(&mut game, 100, 0.6, false);

    let mes_ev = compute_mes_ev(&game);
    for (player, &mes_ev) in mes_ev.iter().enumerate() {
        let br = game.compute_best_response(player, false);
        assert!(
            (br.total_ev() - mes_ev).abs() < 1e-4,
            "player {player}: {} vs {mes_ev}",
            br.total_ev(),
        );
    }

    // the position-aware read works unchanged under bunching
    let br = game.compute_best_response(0, false);
    game.back_to_root();
    game.cache_normalized_weights();
    let ev = game.br_expected_values(&br);
    assert_eq!(ev.len(), game.num_private_hands(0));
    assert!(ev.iter().all(|v| v.is_finite()));
}

#[test]
#[should_panic(expected = "Game is not ready")]
fn best_response_requires_allocated_memory() {
    let card_config = CardConfig {
        range: [
            "AA,KK".parse::<Range>().unwrap(),
            "QQ,JJ".parse::<Range>().unwrap(),
        ],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: card_from_str("As").unwrap(),
        river: card_from_str("2c").unwrap(),
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 60,
        effective_stack: 300,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game.compute_best_response(0, false);
}

#[test]
#[should_panic(expected = "Invalid player")]
fn best_response_rejects_a_bad_player() {
    let game = tiny_river_game();
    game.compute_best_response(2, false);
}

#[test]
#[should_panic(expected = "Storage mode is not compatible")]
fn best_response_rejects_reduced_storage() {
    let game = truncated_turn_game();
    game.compute_best_response(0, false);
}

#[test]
#[should_panic(expected = "Chance node is not allowed")]
fn br_strategy_rejects_a_chance_node() {
    let card_config = CardConfig {
        range: [
            "AA,KK".parse::<Range>().unwrap(),
            "QQ,JJ".parse::<Range>().unwrap(),
        ],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: card_from_str("As").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: 60,
        effective_stack: 60,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game.allocate_memory(false);
    finalize(&mut game);

    let br = game.compute_best_response(0, false);
    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.br_strategy(&br);
}

#[test]
#[should_panic(expected = "Current player is not the best-response player")]
fn br_strategy_rejects_the_wrong_player() {
    let game = tiny_river_game();
    let br = game.compute_best_response(1, false);
    game.br_strategy(&br); // the root is OOP's node
}

#[test]
#[should_panic(expected = "Normalized weights are not cached")]
fn br_expected_values_requires_cached_weights() {
    let mut game = tiny_river_game();
    let br = game.compute_best_response(0, false);
    game.play(0);
    game.br_expected_values(&br);
}

#[test]
#[should_panic(expected = "without detail")]
fn br_expected_values_detail_requires_detail() {
    let mut game = tiny_river_game();
    let br = game.compute_best_response(0, false);
    game.cache_normalized_weights();
    game.br_expected_values_detail(&br);
}

/// A permutation-sum Malmuth–Harville, for cross-checking the subset DP. Exponential — test
/// sizes only.
fn brute_force_icm_equity(payouts: &[f64], stacks: &[f64]) -> Vec<f64> {
    fn recurse(
        payouts: &[f64],
        stacks: &[f64],
        remaining: &[usize],
        place: usize,
        prob: f64,
        equity: &mut [f64],
    ) {
        if place >= payouts.len() || remaining.is_empty() {
            return;
        }
        let total: f64 = remaining.iter().map(|&i| stacks[i]).sum();
        for (pos, &i) in remaining.iter().enumerate() {
            let p = if total > 0.0 {
                prob * stacks[i] / total
            } else {
                prob / remaining.len() as f64
            };
            if p == 0.0 {
                continue;
            }
            equity[i] += payouts[place] * p;
            let mut rest = remaining.to_vec();
            rest.remove(pos);
            recurse(payouts, stacks, &rest, place + 1, p, equity);
        }
    }

    let mut equity = vec![0.0; stacks.len()];
    let players: Vec<usize> = (0..stacks.len()).collect();
    recurse(payouts, stacks, &players, 0, 1.0, &mut equity);
    equity
}

#[test]
fn icm_equity_matches_brute_force_permutations() {
    let cases: &[(&[f64], &[f64])] = &[
        (&[100.0, 60.0], &[50.0, 30.0, 20.0]),
        (&[100.0, 60.0, 40.0], &[500.0, 300.0, 150.0, 50.0]),
        (&[90.0, 50.0, 30.0], &[970.0, 1030.0, 480.0, 2520.0, 310.0]),
        (&[75.0], &[10.0, 20.0, 30.0, 25.0, 15.0]),
        // a busted seat, the shape every all-in terminal produces
        (&[100.0, 60.0, 40.0], &[700.0, 0.0, 300.0]),
        (&[100.0, 60.0], &[1.0, 2.0, 4.0, 8.0, 16.0]),
    ];

    for (payouts, stacks) in cases {
        let dp = icm_equity(payouts, stacks);
        let brute = brute_force_icm_equity(payouts, stacks);
        for (seat, (&a, &b)) in dp.iter().zip(brute.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "payouts {payouts:?}, stacks {stacks:?}, seat {seat}: DP {a} vs brute {b}"
            );
        }
    }
}

/// A heads-up ICM configuration whose numbers are checkable by hand: equity is linear in
/// chips, `eq_i = 60 + 40 * v_i / 2000` at these stacks.
fn icm_heads_up_config() -> IcmConfig {
    IcmConfig {
        payouts: vec![100.0, 60.0],
        stacks: vec![970.0, 970.0],
        oop_seat: 0,
        ip_seat: 1,
    }
}

#[test]
fn icm_two_player_terminal_dollars_are_hand_checkable() {
    // The always-win spot: AA versus a crushed range on AcAdKh, no betting.
    let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), lose_range_str.parse().unwrap()],
        flop: flop_from_str("AcAdKh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game.set_icm_effect(&icm_heads_up_config()).unwrap();

    // By hand: T = 2000 chips, $ per chip = (100 - 60) / 2000 = 0.02.
    // Baseline: both stacks 1000 after the even pot split, so base = 60 + 1000 * 0.02 = 80.
    // OOP always wins the 60-chip pot: 1030 chips = $80.6 absolute, a delta of +$0.6.
    let state = game.icm.as_ref().unwrap();
    assert!((state.base[0] - 80.0).abs() < 1e-9);
    assert!((state.base[1] - 80.0).abs() < 1e-9);
    let payoff = &state.payoffs[&0];
    assert!((payoff[0].win - 0.6).abs() < 1e-9);
    assert!((payoff[0].lose - -0.6).abs() < 1e-9);
    assert!((payoff[1].win - 0.6).abs() < 1e-9);
    assert!((payoff[1].lose - -0.6).abs() < 1e-9);
    // Unraked: the tie vector is the baseline vector, exactly.
    assert!(payoff[0].tie == 0.0 && payoff[1].tie == 0.0);

    game.allocate_memory(false);
    finalize(&mut game);

    let current_ev = compute_current_ev(&game);
    assert!((current_ev[0] - 0.6).abs() < 1e-5);
    assert!((current_ev[1] - -0.6).abs() < 1e-5);
    // Heads up, unraked ICM is exactly zero-sum: equity is linear in chips.
    assert!((current_ev[0] + current_ev[1]).abs() < 1e-6);

    // The display path answers absolute tournament $EV.
    game.cache_normalized_weights();
    let ev_oop = compute_average(&game.expected_values(0), game.normalized_weights(0));
    let ev_ip = compute_average(&game.expected_values(1), game.normalized_weights(1));
    assert!((ev_oop - 80.6).abs() < 1e-4);
    assert!((ev_ip - 79.4).abs() < 1e-4);

    // A chip flip is worth the average: the always-tie spot lands exactly on the baseline.
    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), "AA".parse().unwrap()],
        flop: flop_from_str("2c6dTh").unwrap(),
        ..Default::default()
    };
    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };
    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game.set_icm_effect(&icm_heads_up_config()).unwrap();
    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let ev_oop = compute_average(&game.expected_values(0), game.normalized_weights(0));
    let ev_ip = compute_average(&game.expected_values(1), game.normalized_weights(1));
    assert!((ev_oop - 80.0).abs() < 1e-4);
    assert!((ev_ip - 80.0).abs() < 1e-4);
}

#[test]
fn icm_with_rake_takes_the_three_pass_branch_with_hand_checked_values() {
    // Always-tie on 3-max stacks. (Heads up with equal stacks, symmetric rake would cancel:
    // Malmuth–Harville measures shares of the pool, and burning chips evenly changes no
    // share. A bystander who does not pay the rake is what makes the tie payoff nonzero —
    // and forces the 3-pass showdown branch.)
    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), "AA".parse().unwrap()],
        flop: flop_from_str("2c6dTh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 200,
        effective_stack: 900,
        rake_rate: 0.05,
        rake_cap: 10.0,
        ..Default::default()
    };

    let icm = icm_three_max_config();
    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game.set_icm_effect(&icm).unwrap();

    // By hand: P = 200, rake = min(200 * 0.05, 10) = 10. Baseline stacks [1000, 1500, 2500];
    // a tie at amount 0 leaves [995, 1495, 2500]; a win takes 190 on top of 900.
    let base_equity = icm_equity(&icm.payouts, &[1000.0, 1500.0, 2500.0]);
    let tie_equity = icm_equity(&icm.payouts, &[995.0, 1495.0, 2500.0]);
    let oop_wins_equity = icm_equity(&icm.payouts, &[1090.0, 1400.0, 2500.0]);
    let ip_wins_equity = icm_equity(&icm.payouts, &[900.0, 1590.0, 2500.0]);

    let state = game.icm.as_ref().unwrap();
    let payoff = &state.payoffs[&0];
    assert!(
        payoff[0].tie != 0.0,
        "rake must make the tie payoff nonzero"
    );
    assert!((payoff[0].tie - (tie_equity[0] - base_equity[0])).abs() < 1e-12);
    assert!((payoff[1].tie - (tie_equity[1] - base_equity[1])).abs() < 1e-12);
    assert!((payoff[0].win - (oop_wins_equity[0] - base_equity[0])).abs() < 1e-12);
    assert!((payoff[0].lose - (ip_wins_equity[0] - base_equity[0])).abs() < 1e-12);
    assert!((payoff[1].win - (ip_wins_equity[1] - base_equity[1])).abs() < 1e-12);
    assert!((payoff[1].lose - (oop_wins_equity[1] - base_equity[1])).abs() < 1e-12);

    game.allocate_memory(false);
    finalize(&mut game);

    // Every showdown ties, so the absolute $EV is exactly the tie equity.
    game.cache_normalized_weights();
    let ev_oop = compute_average(&game.expected_values(0), game.normalized_weights(0));
    let ev_ip = compute_average(&game.expected_values(1), game.normalized_weights(1));
    assert!((ev_oop - tie_equity[0] as f32).abs() < 1e-4);
    assert!((ev_ip - tie_equity[1] as f32).abs() < 1e-4);
}

/// The spec's 3-max situation: two short-ish contestants and a big-stacked bystander.
fn icm_three_max_config() -> IcmConfig {
    IcmConfig {
        payouts: vec![50.0, 30.0, 20.0],
        stacks: vec![900.0, 1400.0, 2500.0],
        oop_seat: 0,
        ip_seat: 1,
    }
}

/// A hand-enumerable river game on the 3-max ICM stacks (pot 200, effective stack 900).
fn icm_river_game() -> PostFlopGame {
    let card_config = CardConfig {
        range: [
            "AA,KK,QQ".parse::<Range>().unwrap(),
            "JJ,TT".parse::<Range>().unwrap(),
        ],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: card_from_str("As").unwrap(),
        river: card_from_str("2c").unwrap(),
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 200,
        effective_stack: 900,
        river_bet_sizes: [
            BetSizeOptions::try_from(("50%", "3x")).unwrap(),
            BetSizeOptions::try_from(("50%", "3x")).unwrap(),
        ],
        add_allin_threshold: 1.5,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    PostFlopGame::with_config(card_config, action_tree).unwrap()
}

#[test]
fn icm_terminal_payoffs_conserve_the_prize_pool() {
    // The identity that replaces zero-sum: at every terminal event, the $ the contestants
    // gain or lose relative to their baselines is exactly offset by the bystanders' equity
    // shift — Malmuth–Harville conserves the prize pool. Heads up there is no bystander and
    // the game is zero-sum; with one, the contestants' deltas sum to minus the bystander's.
    let icm = icm_three_max_config();
    let mut game = icm_river_game();
    game.set_icm_effect(&icm).unwrap();

    let state = game.icm.as_ref().unwrap();
    let starting_pot = 200.0;
    let bystander = 2;

    let mut base_stacks = icm.stacks.clone();
    base_stacks[icm.oop_seat] += 0.5 * starting_pot;
    base_stacks[icm.ip_seat] += 0.5 * starting_pot;
    let base_equity = icm_equity(&icm.payouts, &base_stacks);

    assert!(!state.payoffs.is_empty());
    for (&amount, payoff) in &state.payoffs {
        let bet = amount as f64;
        let pot = starting_pot + 2.0 * bet;

        // Unraked ICM keeps the exact-zero tie payoff (and with it the 2-pass showdown).
        assert!(payoff[0].tie == 0.0 && payoff[1].tie == 0.0);

        let mut oop_wins = icm.stacks.clone();
        oop_wins[icm.oop_seat] += pot - bet;
        oop_wins[icm.ip_seat] -= bet;
        let oop_wins_equity = icm_equity(&icm.payouts, &oop_wins);

        let bystander_delta = oop_wins_equity[bystander] - base_equity[bystander];
        let contestant_delta = payoff[0].win + payoff[1].lose;
        assert!(
            (contestant_delta + bystander_delta).abs() < 1e-9,
            "amount {amount}: contestants {contestant_delta} vs bystander {bystander_delta}"
        );
        // And the bystander genuinely moves: an all-in bust hands them real equity, which is
        // what makes the game non-zero-sum for the contestants.
        if amount == 900 {
            assert!(bystander_delta.abs() > 0.1);
        }
    }
}

#[test]
fn icm_solve_is_the_chip_solve_under_linear_payoffs() {
    // Heads up with the prize gap equal to the chips in play, a dollar is a chip: the solve
    // must reproduce the chip-mode strategy, and every $ number is the chip number scaled by
    // (p1 - p2) / T. Here the scale is kept at 1 to make the comparison exact.
    let total_chips = 300.0 + 300.0 + 60.0;
    let icm = IcmConfig {
        payouts: vec![total_chips, 0.0],
        stacks: vec![300.0, 300.0],
        oop_seat: 0,
        ip_seat: 1,
    };

    let mut chip_game = tiny_river_game_unsolved();
    chip_game.allocate_memory(false);
    solve(&mut chip_game, 200, 0.0, false);

    let mut icm_game = tiny_river_game_unsolved();
    icm_game.set_icm_effect(&icm).unwrap();
    icm_game.allocate_memory(false);
    solve(&mut icm_game, 200, 0.0, false);

    // Same strategy at the root (regret matching is invariant under utility scaling, and the
    // scale is 1 anyway)...
    let chip_strategy = chip_game.strategy();
    let icm_strategy = icm_game.strategy();
    for (a, b) in chip_strategy.iter().zip(icm_strategy.iter()) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }

    // ... the same root EV deltas (the ICM baseline is T/2 + stack = everyone's chip stake,
    // so deltas coincide with the chip bias convention) ...
    let chip_ev = compute_current_ev(&chip_game);
    let icm_ev = compute_current_ev(&icm_game);
    assert!(
        (chip_ev[0] - icm_ev[0]).abs() < 1e-3,
        "{chip_ev:?} vs {icm_ev:?}"
    );
    assert!((chip_ev[1] - icm_ev[1]).abs() < 1e-3);

    // ... and the same exploitability, though it flows through the current-EV branch.
    let chip_exploitability = compute_exploitability(&chip_game);
    let icm_exploitability = compute_exploitability(&icm_game);
    assert!((chip_exploitability - icm_exploitability).abs() < 1e-3);

    // The pot-value helper agrees that a chip is a dollar here.
    assert!((icm_pot_value(&icm, 60) - 60.0).abs() < 1e-9);
}

#[test]
fn icm_exploitability_converges_through_the_current_ev_branch() {
    let mut game = icm_river_game();
    game.set_icm_effect(&icm_three_max_config()).unwrap();
    game.allocate_memory(false);

    let pot_value = icm_pot_value(&icm_three_max_config(), 200);
    assert!(pot_value > 0.0);

    solve(&mut game, 2000, (pot_value * 0.001) as f32, false);

    // The game is genuinely non-zero-sum in $: the current EVs do not cancel...
    let current_ev = compute_current_ev(&game);
    assert!(
        (current_ev[0] + current_ev[1]).abs() > 1e-4,
        "expected a non-zero-sum game, got {current_ev:?}"
    );

    // ... so the naive (mes0 + mes1) / 2 is broken here, and the raked-style branch is the
    // one that measures convergence. It converged.
    let exploitability = compute_exploitability(&game);
    assert!(
        exploitability <= (pot_value * 0.001) as f32,
        "exploitability {exploitability} did not reach 0.1% of the pot value {pot_value}"
    );

    let mes_ev = compute_mes_ev(&game);
    let naive = (mes_ev[0] + mes_ev[1]) * 0.5;
    let honest = ((mes_ev[0] - current_ev[0]) + (mes_ev[1] - current_ev[1])) * 0.5;
    assert!((exploitability - honest).abs() < 1e-6);
    assert!(
        (naive - honest).abs() > 1e-4,
        "naive {naive} vs honest {honest}"
    );
}

#[test]
fn icm_expected_values_are_absolute_dollars() {
    let mut game = icm_river_game();
    game.set_icm_effect(&icm_three_max_config()).unwrap();
    game.allocate_memory(false);
    solve(&mut game, 500, 0.005, false);

    let current_ev = compute_current_ev(&game);
    let base = {
        let state = game.icm.as_ref().unwrap();
        state.base
    };

    game.cache_normalized_weights();
    for player in 0..2 {
        // The root range-weighted average of the absolute $ EVs is the $ delta plus the
        // baseline — the display path and the solver-facing path agree.
        let ev = compute_average(
            &game.expected_values(player),
            game.normalized_weights(player),
        );
        let expected = current_ev[player] + base[player] as f32;
        assert!(
            (ev - expected).abs() < 1e-3,
            "player {player}: {ev} vs {expected}"
        );

        // expected_values is the strategy-weighted average of expected_values_detail.
        if player == game.current_player() {
            let detail = game.expected_values_detail(player);
            let strategy = game.strategy();
            let num_hands = game.num_private_hands(player);
            let ev_by_hand = game.expected_values(player);
            for hand in 0..num_hands {
                let mut avg = 0.0;
                for action in 0..detail.len() / num_hands {
                    avg += detail[action * num_hands + hand] * strategy[action * num_hands + hand];
                }
                assert!((avg - ev_by_hand[hand]).abs() < 1e-2);
            }
        }
    }
}

/// A view of a solved game that `finalize` accepts again — the engine-side mirror of the
/// sidecar's `Refinalize` wrapper, needed because the decoder restores stored EVs with plain
/// chip evaluation before any runtime effect can be re-applied.
struct Refinalize<'a>(&'a mut PostFlopGame);

impl Game for Refinalize<'_> {
    type Node = PostFlopNode;

    fn root(&self) -> MutexGuardLike<'_, Self::Node> {
        self.0.root()
    }

    fn num_private_hands(&self, player: usize) -> usize {
        self.0.num_private_hands(player)
    }

    fn initial_weights(&self, player: usize) -> &[f32] {
        self.0.initial_weights(player)
    }

    fn evaluate(
        &self,
        result: &mut [std::mem::MaybeUninit<f32>],
        node: &Self::Node,
        player: usize,
        cfreach: &[f32],
    ) {
        self.0.evaluate(result, node, player, cfreach);
    }

    fn chance_factor(&self, node: &Self::Node) -> usize {
        self.0.chance_factor(node)
    }

    fn is_solved(&self) -> bool {
        false
    }

    fn set_solved(&mut self) {}

    fn is_ready(&self) -> bool {
        true
    }

    fn is_raked(&self) -> bool {
        self.0.is_raked()
    }

    fn is_icm(&self) -> bool {
        self.0.is_icm()
    }

    fn isomorphic_chances(&self, node: &Self::Node) -> &[u8] {
        self.0.isomorphic_chances(node)
    }

    fn isomorphic_swap(&self, node: &Self::Node, index: usize) -> &[Vec<(u16, u16)>; 2] {
        self.0.isomorphic_swap(node, index)
    }

    fn locking_strategy(&self, node: &Self::Node) -> &[f32] {
        self.0.locking_strategy(node)
    }

    fn is_compression_enabled(&self) -> bool {
        self.0.is_compression_enabled()
    }
}

/// `tiny_river_game` without the allocate-and-finalize tail, for tests that want to install
/// effects first.
fn tiny_river_game_unsolved() -> PostFlopGame {
    let card_config = CardConfig {
        range: [
            "AA,KK".parse::<Range>().unwrap(),
            "QQ,JJ".parse::<Range>().unwrap(),
        ],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: card_from_str("As").unwrap(),
        river: card_from_str("2c").unwrap(),
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 60,
        effective_stack: 300,
        river_bet_sizes: [
            BetSizeOptions::try_from(("50%", "")).unwrap(),
            BetSizeOptions::try_from(("50%", "")).unwrap(),
        ],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    PostFlopGame::with_config(card_config, action_tree).unwrap()
}

/// Root and one-action-in observations that must survive a save/load/reapply round trip.
fn icm_observations(game: &mut PostFlopGame) -> (Vec<f32>, Vec<f32>, Vec<f32>, [f32; 2]) {
    game.back_to_root();
    game.cache_normalized_weights();
    let root_ev = game.expected_values(game.current_player());
    let root_strategy = game.strategy();
    game.play(0);
    game.cache_normalized_weights();
    let next_ev = game.expected_values(game.current_player());
    game.back_to_root();
    (root_ev, root_strategy, next_ev, compute_current_ev(&*game))
}

#[test]
fn icm_solve_survives_save_load_reapply_bit_for_bit() {
    let icm = icm_three_max_config();
    let mut game = icm_river_game();
    game.set_icm_effect(&icm).unwrap();
    game.allocate_memory(false);
    solve(&mut game, 300, 0.001, false);

    let before = icm_observations(&mut game);

    let mut buffer = Vec::new();
    save_data_into_std_write(&game, "", &mut buffer, None).unwrap();
    let mut cursor = std::io::Cursor::new(&buffer);
    let mut loaded: PostFlopGame = load_data_from_std_read(&mut cursor, None).unwrap().0;

    // The file recorded nothing about ICM: the decoder restored the stored EVs with chip
    // evaluation, so the reapply is set_icm_effect *plus* one re-finalize.
    assert!(loaded.icm_config().is_none());
    loaded.set_icm_effect(&icm).unwrap();
    finalize(&mut Refinalize(&mut loaded));

    let after = icm_observations(&mut loaded);
    assert_eq!(before.0, after.0, "root EVs changed across the round trip");
    assert_eq!(
        before.1, after.1,
        "the strategy changed across the round trip"
    );
    assert_eq!(before.2, after.2, "child EVs changed across the round trip");
    assert_eq!(
        before.3, after.3,
        "current EV changed across the round trip"
    );
}

#[test]
fn icm_composes_with_bunching() {
    let icm = icm_heads_up_config();
    let flop = flop_from_str("Td9d6h").unwrap();

    let build = || {
        let card_config = CardConfig {
            range: [
                "AA,KK".parse::<Range>().unwrap(),
                "QQ,JJ".parse::<Range>().unwrap(),
            ],
            flop,
            turn: card_from_str("As").unwrap(),
            river: card_from_str("2c").unwrap(),
        };
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 60,
            effective_stack: 970,
            river_bet_sizes: [
                BetSizeOptions::try_from(("50%", "")).unwrap(),
                BetSizeOptions::try_from(("50%", "")).unwrap(),
            ],
            ..Default::default()
        };
        let action_tree = ActionTree::new(tree_config).unwrap();
        PostFlopGame::with_config(card_config, action_tree).unwrap()
    };

    let mut bunching_data = BunchingData::new(&["JJ+".parse().unwrap()], flop).unwrap();
    bunching_data.process(false);

    // Chip-mode bunching solve as the reference; the two effects are independent, and heads-up
    // linear payouts make the ICM answer the chip answer.
    let mut chip_game = build();
    chip_game.set_bunching_effect(&bunching_data).unwrap();
    chip_game.allocate_memory(false);
    solve(&mut chip_game, 200, 0.0, false);

    let linear_icm = IcmConfig {
        payouts: vec![970.0 + 970.0 + 60.0, 0.0],
        stacks: vec![970.0, 970.0],
        oop_seat: 0,
        ip_seat: 1,
    };
    let mut both_game = build();
    both_game.set_bunching_effect(&bunching_data).unwrap();
    both_game.set_icm_effect(&linear_icm).unwrap();
    assert!(both_game.is_icm());
    both_game.allocate_memory(false);
    solve(&mut both_game, 200, 0.0, false);

    let chip_ev = compute_current_ev(&chip_game);
    let both_ev = compute_current_ev(&both_game);
    assert!(
        (chip_ev[0] - both_ev[0]).abs() < 1e-2,
        "{chip_ev:?} vs {both_ev:?}"
    );
    assert!((chip_ev[1] - both_ev[1]).abs() < 1e-2);
    for (a, b) in chip_game.strategy().iter().zip(both_game.strategy().iter()) {
        assert!((a - b).abs() < 1e-3);
    }

    // The round trip with BOTH effects re-applied, one re-finalize at the end.
    let before = icm_observations(&mut both_game);
    let mut buffer = Vec::new();
    save_data_into_std_write(&both_game, "", &mut buffer, None).unwrap();
    let mut cursor = std::io::Cursor::new(&buffer);
    let mut loaded: PostFlopGame = load_data_from_std_read(&mut cursor, None).unwrap().0;
    loaded.set_bunching_effect(&bunching_data).unwrap();
    loaded.set_icm_effect(&linear_icm).unwrap();
    finalize(&mut Refinalize(&mut loaded));
    let after = icm_observations(&mut loaded);
    assert_eq!(before.0, after.0);
    assert_eq!(before.1, after.1);
    assert_eq!(before.2, after.2);
    assert_eq!(before.3, after.3);
    let _ = icm; // the heads-up helper is exercised by the other tests
}

#[test]
fn icm_effect_lifecycle_mirrors_bunching() {
    let mut game = icm_river_game();
    assert!(game.icm_config().is_none());
    assert!(!game.is_icm());

    game.set_icm_effect(&icm_three_max_config()).unwrap();
    assert!(game.is_icm());
    assert_eq!(game.icm_config(), Some(&icm_three_max_config()));

    // A mismatched configuration is refused before anything is touched: the previous effect
    // stays in place.
    let mut wrong = icm_three_max_config();
    wrong.stacks[0] = 800.0;
    let err = game.set_icm_effect(&wrong).unwrap_err();
    assert!(err.contains("effective stack"), "{err}");
    assert_eq!(game.icm_config(), Some(&icm_three_max_config()));

    game.reset_icm_effect();
    assert!(!game.is_icm());

    // update_config resets the effect, like the bunching lifecycle.
    game.set_icm_effect(&icm_three_max_config()).unwrap();
    let fresh = icm_river_game();
    let tree_config = fresh.tree_config().clone();
    let card_config = fresh.card_config().clone();
    game.update_config(card_config, ActionTree::new(tree_config).unwrap())
        .unwrap();
    assert!(game.icm_config().is_none());

    // An uninitialized game refuses the effect.
    let mut uninitialized = PostFlopGame::new();
    let err = uninitialized
        .set_icm_effect(&icm_three_max_config())
        .unwrap_err();
    assert!(err.contains("not successfully initialized"), "{err}");
}
