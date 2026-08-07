use super::*;
use crate::interface::*;
use crate::sliceop::*;
use crate::utility::*;
use std::mem::MaybeUninit;
use std::sync::Mutex;

#[cfg(feature = "custom-alloc")]
use crate::alloc::*;

/// A snapshot of one player's best response (maximally exploitative strategy) against the
/// opponent's current strategy.
///
/// Produced by [`PostFlopGame::compute_best_response`] and read back through the position-aware
/// accessors ([`br_strategy`], [`br_actions`], [`br_expected_values`], and
/// [`br_expected_values_detail`]), which interpret the snapshot at the game's current node with
/// the same conventions as [`strategy`] and [`expected_values`].
///
/// The data is keyed by node and stored in storage coordinates (like locking strategies), so a
/// snapshot stays valid wherever the game is navigated afterwards. It is never serialized.
///
/// [`br_strategy`]: PostFlopGame::br_strategy
/// [`br_actions`]: PostFlopGame::br_actions
/// [`br_expected_values`]: PostFlopGame::br_expected_values
/// [`br_expected_values_detail`]: PostFlopGame::br_expected_values_detail
/// [`strategy`]: PostFlopGame::strategy
/// [`expected_values`]: PostFlopGame::expected_values
pub struct BestResponse {
    player: usize,
    nodes: BTreeMap<usize, BestResponseNode>,
    total_ev: f32,
    keep_detail: bool,
    arena_len: usize,
    num_hands: usize,
}

/// Per-node data of a [`BestResponse`], in storage coordinates.
struct BestResponseNode {
    /// Per-hand best-response counterfactual values; the same cfreach convention as the stored
    /// cfvalues.
    values: Vec<f32>,
    /// Argmax action per hand at the best-response player's multi-action decision nodes; ties
    /// are broken to the lowest action index, and locked hands hold `u16::MAX`.
    actions: Option<Vec<u16>>,
    /// The full per-action value block (`#(actions) * #(private hands)`, action-major), kept
    /// only when requested.
    detail: Option<Vec<f32>>,
}

impl BestResponse {
    /// Returns the player this best response was computed for (0 = OOP, 1 = IP).
    #[inline]
    pub fn player(&self) -> usize {
        self.player
    }

    /// Returns the whole-game expected value of the best response.
    ///
    /// The bias, i.e., (starting pot) / 2, is already subtracted, so the value follows the same
    /// convention as [`compute_mes_ev`] and is directly comparable to [`compute_current_ev`].
    ///
    /// [`compute_mes_ev`]: crate::compute_mes_ev
    /// [`compute_current_ev`]: crate::compute_current_ev
    #[inline]
    pub fn total_ev(&self) -> f32 {
        self.total_ev
    }

    /// Returns whether per-action values were kept (see
    /// [`PostFlopGame::br_expected_values_detail`]).
    #[inline]
    pub fn has_detail(&self) -> bool {
        self.keep_detail
    }

    /// Returns the estimated memory usage of this snapshot in bytes.
    #[inline]
    pub fn memory_usage(&self) -> u64 {
        let mut ret = 0;
        for node in self.nodes.values() {
            ret += vec_memory_usage(&node.values);
            if let Some(actions) = &node.actions {
                ret += vec_memory_usage(actions);
            }
            if let Some(detail) = &node.detail {
                ret += vec_memory_usage(detail);
            }
        }
        ret
    }

    /// Panics unless this snapshot was computed against a game of the same shape.
    #[inline]
    fn check_fingerprint(&self, game: &PostFlopGame) {
        if self.arena_len != game.node_arena.len()
            || self.num_hands != game.num_private_hands(self.player)
        {
            panic!("Best response does not belong to this game");
        }
    }
}

impl PostFlopGame {
    /// Computes the best response of the given player against the opponent's current strategy.
    ///
    /// The returned snapshot holds per-hand best-response values at every non-terminal node,
    /// and the argmax actions (plus, if `keep_detail` is set, the per-action values) at the
    /// player's own decision nodes. Locked nodes are respected: a locked hand keeps its pinned
    /// distribution rather than a pure argmax.
    ///
    /// The computation is position-independent and does not move the current node. Its total EV
    /// equals `compute_mes_ev(game)[player]` (it is the same recursion).
    ///
    /// Panics if the memory is not allocated, the game was loaded with a reduced storage mode,
    /// or `player` is not 0 or 1.
    pub fn compute_best_response(&self, player: usize, keep_detail: bool) -> BestResponse {
        if !self.is_ready() && !self.is_solved() {
            panic!("Game is not ready");
        }

        // `is_solved` alone does not imply full storage on a loaded game.
        if self.storage_mode != BoardState::River {
            panic!("Storage mode is not compatible");
        }

        if player > 1 {
            panic!("Invalid player");
        }

        let num_hands = self.num_private_hands(player);
        let sink = Mutex::new(BTreeMap::new());

        let mut result = Vec::with_capacity(num_hands);
        best_response_recursive(
            result.spare_capacity_mut(),
            self,
            &self.root(),
            player,
            self.initial_weights(player ^ 1),
            &sink,
            keep_detail,
        );
        unsafe { result.set_len(num_hands) };

        let total_ev = weighted_sum(&result, self.initial_weights(player));

        BestResponse {
            player,
            nodes: sink.into_inner().unwrap(),
            total_ev,
            keep_detail,
            arena_len: self.node_arena.len(),
            num_hands,
        }
    }

    /// Returns the pure best-response strategy of `br`'s player at the current node.
    ///
    /// The return value has the same length and layout as [`strategy`]:
    /// `#(actions) * #(private hands)`, action-major. An unlocked hand holds 1.0 at its argmax
    /// action and 0.0 elsewhere; a locked hand holds its pinned distribution.
    ///
    /// Panics if the memory is not allocated, the current node is a terminal or chance node, or
    /// the current player is not `br`'s player.
    ///
    /// [`strategy`]: #method.strategy
    pub fn br_strategy(&self, br: &BestResponse) -> Vec<f32> {
        let node_index = self.br_hero_node_index(br);
        let player = br.player;
        let num_hands = br.num_hands;
        let num_actions = self.node_arena[node_index].lock().num_actions();

        if num_actions == 1 {
            return vec![1.0; num_hands];
        }

        let actions = br.nodes[&node_index].actions.as_ref().unwrap();
        let locking = self.locking_strategy.get(&node_index);
        let mut ret = vec![0.0; num_actions * num_hands];

        for hand in 0..num_hands {
            let action = actions[hand];
            if action == u16::MAX {
                let locking = locking.unwrap();
                for action in 0..num_actions {
                    ret[action * num_hands + hand] = locking[action * num_hands + hand];
                }
            } else {
                ret[action as usize * num_hands + hand] = 1.0;
            }
        }

        ret.chunks_exact_mut(num_hands).for_each(|chunk| {
            self.apply_swap(chunk, player, false);
        });

        ret
    }

    /// Returns the per-hand argmax action of `br`'s player at the current node, as indices into
    /// [`available_actions`]. A locked hand answers `None` (it keeps its pinned distribution).
    ///
    /// Panics as [`br_strategy`].
    ///
    /// [`available_actions`]: #method.available_actions
    /// [`br_strategy`]: #method.br_strategy
    pub fn br_actions(&self, br: &BestResponse) -> Vec<Option<usize>> {
        let node_index = self.br_hero_node_index(br);
        let num_actions = self.node_arena[node_index].lock().num_actions();

        if num_actions == 1 {
            return vec![Some(0); br.num_hands];
        }

        let mut actions = br.nodes[&node_index].actions.clone().unwrap();
        self.apply_swap(&mut actions, br.player, false);

        actions
            .iter()
            .map(|&action| (action != u16::MAX).then_some(action as usize))
            .collect()
    }

    /// Returns the expected value of each private hand of `br`'s player at the current node,
    /// assuming best-response play from here on.
    ///
    /// The units and normalization are those of [`expected_values`], and the value is defined at
    /// every non-terminal node — chance nodes and the opponent's decision nodes included. Hands
    /// with zero normalized weight answer 0.0.
    ///
    /// Panics if the game is neither ready nor solved, the current node is a terminal node, or
    /// the normalized weights are not cached.
    ///
    /// [`expected_values`]: #method.expected_values
    pub fn br_expected_values(&self, br: &BestResponse) -> Vec<f32> {
        if !self.is_ready() && !self.is_solved() {
            panic!("Game is not ready");
        }

        if self.is_terminal_node() {
            panic!("Terminal node is not allowed");
        }

        if !self.is_normalized_weight_cached {
            panic!("Normalized weights are not cached");
        }

        br.check_fingerprint(self);

        let mut ret = br.nodes[&self.current_node_index()].values.clone();
        self.br_normalize(&mut ret, br.player, false);
        ret
    }

    /// Returns the expected value of each action of each private hand of `br`'s player at the
    /// current node, assuming best-response play below each action.
    ///
    /// The layout and conventions are those of [`expected_values_detail`] at the player's own
    /// decision node: `#(actions) * #(private hands)`, action-major, with `Fold` rows forced to
    /// 0.0 and zero-weight hands 0.0.
    ///
    /// Panics as [`br_expected_values`]; additionally if the current node is a chance node or
    /// not `br`'s player's decision node, or if the best response was computed without
    /// `keep_detail`.
    ///
    /// [`expected_values_detail`]: #method.expected_values_detail
    /// [`br_expected_values`]: #method.br_expected_values
    pub fn br_expected_values_detail(&self, br: &BestResponse) -> Vec<f32> {
        if !self.is_ready() && !self.is_solved() {
            panic!("Game is not ready");
        }

        if !br.keep_detail {
            panic!("Best response was computed without detail");
        }

        let node_index = self.br_hero_node_index(br);

        if !self.is_normalized_weight_cached {
            panic!("Normalized weights are not cached");
        }

        let entry = &br.nodes[&node_index];
        // A single-action node short-circuits the recursion, so only the pass-through values
        // exist; they are exactly the one action's values.
        let mut ret = match &entry.detail {
            Some(detail) => detail.clone(),
            None => entry.values.clone(),
        };

        self.br_normalize(&mut ret, br.player, true);
        ret
    }

    /// Returns the arena index of the current node.
    #[inline]
    fn current_node_index(&self) -> usize {
        self.node_history.last().copied().unwrap_or(0)
    }

    /// The shared guard of [`br_strategy`], [`br_actions`], and [`br_expected_values_detail`]:
    /// panics unless the current node is `br`'s player's decision node, and returns its index.
    ///
    /// [`br_strategy`]: #method.br_strategy
    /// [`br_actions`]: #method.br_actions
    /// [`br_expected_values_detail`]: #method.br_expected_values_detail
    fn br_hero_node_index(&self, br: &BestResponse) -> usize {
        if self.state < State::MemoryAllocated {
            panic!("Memory is not allocated");
        }

        if self.is_terminal_node() {
            panic!("Terminal node is not allowed");
        }

        if self.is_chance_node() {
            panic!("Chance node is not allowed");
        }

        if self.current_player() != br.player {
            panic!("Current player is not the best-response player");
        }

        br.check_fingerprint(self);
        self.current_node_index()
    }

    /// Converts raw best-response counterfactual values (one or more action-major rows in
    /// storage coordinates) into the display convention of [`expected_values_detail`].
    ///
    /// [`expected_values_detail`]: #method.expected_values_detail
    fn br_normalize(&self, values: &mut [f32], player: usize, have_actions: bool) {
        let num_hands = self.num_private_hands(player);

        let mut chance_factor = 1;
        if self.card_config.turn == NOT_DEALT && self.turn != NOT_DEALT {
            chance_factor *= 45 - self.bunching_num_dead_cards;
        }
        if self.card_config.river == NOT_DEALT && self.river != NOT_DEALT {
            chance_factor *= 44 - self.bunching_num_dead_cards;
        }

        let num_combinations = match self.bunching_num_dead_cards {
            0 => self.num_combinations,
            _ => self.bunching_num_combinations,
        };

        let normalizer = (num_combinations * chance_factor as f64) as f32;

        let node = self.node_arena[self.current_node_index()].lock();
        let starting_pot = self.tree_config.starting_pot;
        let total_bet_amount = self.total_bet_amount();
        let bias = (total_bet_amount[player] - total_bet_amount[player ^ 1]).max(0);

        // The same de-bias convention as `expected_values_detail`: the chip "expected pot
        // recovery" addend in chip mode, the absolute $ baseline under ICM.
        let addend = match &self.icm {
            Some(state) => state.base[player] as f32,
            None => starting_pot as f32 * 0.5 + (node.amount + bias) as f32,
        };

        values
            .chunks_exact_mut(num_hands)
            .enumerate()
            .for_each(|(action, row)| {
                let is_fold = have_actions && node.play(action).prev_action == Action::Fold;
                self.apply_swap(row, player, false);
                row.iter_mut()
                    .zip(self.weights[player].iter())
                    .zip(self.normalized_weights[player].iter())
                    .for_each(|((v, &w_raw), &w_normalized)| {
                        if is_fold || w_normalized == 0.0 {
                            *v = 0.0;
                        } else {
                            *v *= normalizer * (w_raw / w_normalized);
                            *v += addend;
                        }
                    });
            });
    }
}

/// The recursive helper for [`PostFlopGame::compute_best_response`].
///
/// A faithful mirror of `compute_best_cfv_recursive` (see `utility.rs`) whose only addition is
/// the sink: after `result` is written, the node's data is recorded keyed by its arena index, in
/// storage coordinates. Everything stored is copied to the global allocator (`to_vec`), which is
/// mandatory under the `custom-alloc` feature — the recursion's buffers must never escape it.
fn best_response_recursive(
    result: &mut [MaybeUninit<f32>],
    game: &PostFlopGame,
    node: &PostFlopNode,
    player: usize,
    cfreach: &[f32],
    sink: &Mutex<BTreeMap<usize, BestResponseNode>>,
    keep_detail: bool,
) {
    // terminal node
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = game.num_private_hands(player);

    // simply recurse when the number of actions is one
    if num_actions == 1 && !node.is_chance() {
        let child = &node.play(0);
        best_response_recursive(result, game, child, player, cfreach, sink, keep_detail);
        let values = unsafe { &*(result as *const _ as *const [f32]) };
        sink.lock().unwrap().insert(
            game.node_index(node),
            BestResponseNode {
                values: values.to_vec(),
                actions: None,
                detail: None,
            },
        );
        return;
    }

    // allocate memory for storing the counterfactual values
    #[cfg(feature = "custom-alloc")]
    let cfv_actions = MutexLike::new(Vec::with_capacity_in(num_actions * num_hands, StackAlloc));
    #[cfg(not(feature = "custom-alloc"))]
    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    // chance node
    if node.is_chance() {
        // update the reach probabilities
        #[cfg(feature = "custom-alloc")]
        let mut cfreach_updated = Vec::with_capacity_in(cfreach.len(), StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach_updated = Vec::with_capacity(cfreach.len());
        mul_slice_scalar_uninit(
            cfreach_updated.spare_capacity_mut(),
            cfreach,
            1.0 / game.chance_factor(node) as f32,
        );
        unsafe { cfreach_updated.set_len(cfreach.len()) };

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            best_response_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                &cfreach_updated,
                sink,
                keep_detail,
            )
        });

        // use 64-bit floating point values
        #[cfg(feature = "custom-alloc")]
        let mut result_f64 = Vec::with_capacity_in(num_hands, StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut result_f64 = Vec::with_capacity(num_hands);

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_f64_uninit(result_f64.spare_capacity_mut(), &cfv_actions);
        unsafe { result_f64.set_len(num_hands) };

        // get information about isomorphic chances
        let isomorphic_chances = game.isomorphic_chances(node);

        // process isomorphic chances
        for (i, &isomorphic_index) in isomorphic_chances.iter().enumerate() {
            let swap_list = &game.isomorphic_swap(node, i)[player];
            let tmp = row_mut(&mut cfv_actions, isomorphic_index as usize, num_hands);

            apply_swap(tmp, swap_list);

            result_f64.iter_mut().zip(&*tmp).for_each(|(r, &v)| {
                *r += v as f64;
            });

            apply_swap(tmp, swap_list);
        }

        result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
            r.write(v as f32);
        });

        let values = unsafe { &*(result as *const _ as *const [f32]) };
        sink.lock().unwrap().insert(
            game.node_index(node),
            BestResponseNode {
                values: values.to_vec(),
                actions: None,
                detail: None,
            },
        );
    }
    // player node
    else if node.player() == player {
        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            best_response_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                cfreach,
                sink,
                keep_detail,
            )
        });

        let locking = game.locking_strategy(node);
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };

        if locking.is_empty() {
            // compute element-wise maximum (take the best response)
            max_slices_uninit(result, &cfv_actions);
        } else {
            // when the node is locked
            max_fma_slices_uninit(result, &cfv_actions, locking);
        }

        // the argmax per hand, ties broken to the lowest action index (`max_slices_uninit`
        // itself keeps the last tied action, so it cannot be inferred from the maximum)
        let mut actions = Vec::with_capacity(num_hands);
        for hand in 0..num_hands {
            // lockedness is uniform across actions, so the first row decides
            if !locking.is_empty() && locking[hand].is_sign_positive() {
                actions.push(u16::MAX);
            } else {
                let mut best_action = 0;
                let mut best_value = cfv_actions[hand];
                for action in 1..num_actions {
                    let value = cfv_actions[action * num_hands + hand];
                    if value > best_value {
                        best_action = action;
                        best_value = value;
                    }
                }
                actions.push(best_action as u16);
            }
        }

        let values = unsafe { &*(result as *const _ as *const [f32]) };
        sink.lock().unwrap().insert(
            game.node_index(node),
            BestResponseNode {
                values: values.to_vec(),
                actions: Some(actions),
                detail: keep_detail.then(|| cfv_actions[..].to_vec()),
            },
        );
    }
    // opponent node
    else {
        // obtain the strategy
        #[cfg(feature = "custom-alloc")]
        let mut cfreach_actions = if game.is_compression_enabled() {
            normalized_strategy_compressed_custom_alloc(node.strategy_compressed(), num_actions)
        } else {
            normalized_strategy_custom_alloc(node.strategy(), num_actions)
        };
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach_actions = if game.is_compression_enabled() {
            normalized_strategy_compressed(node.strategy_compressed(), num_actions)
        } else {
            normalized_strategy(node.strategy(), num_actions)
        };

        // node-locking
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut cfreach_actions, locking);

        // update the reach probabilities
        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            best_response_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                sink,
                keep_detail,
            );
        });

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);

        let values = unsafe { &*(result as *const _ as *const [f32]) };
        sink.lock().unwrap().insert(
            game.node_index(node),
            BestResponseNode {
                values: values.to_vec(),
                actions: None,
                detail: None,
            },
        );
    }
}
