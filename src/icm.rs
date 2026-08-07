//! ICM (Independent Chip Model) support: payout-adjusted solving.
//!
//! Tournament chips are not worth their face value: doubling a stack less than doubles its
//! share of the prize pool. This module maps chip outcomes at the terminals of a postflop game
//! to tournament $EV via the Malmuth–Harville model, so the solver optimizes dollars instead of
//! chips. The mapping is a **runtime effect** in the spirit of the bunching effect:
//!
//! - [`IcmConfig`] describes the tournament situation (remaining prizes, remaining stacks, and
//!   which two seats are contesting the pot).
//! - [`PostFlopGame::set_icm_effect`] precomputes, for every distinct terminal chip amount, the
//!   three $ payoffs (win / tie / lose) of each contestant, expressed as deltas from a
//!   per-player baseline (the street-start stacks with the pot split evenly). The terminal
//!   evaluation then substitutes these three scalars for the chip payoffs — the per-hand loops
//!   are untouched.
//! - The configuration is **never serialized**: a saved file records nothing about ICM, and a
//!   loaded game must have the effect re-applied (see [`PostFlopGame::set_icm_effect`]).
//!
//! # Non-zero-sum caveat
//!
//! An ICM game is not zero-sum in $ — chips lost by one contestant are partly "won" by the
//! bystanders' equities — exactly as a raked game is not zero-sum in chips. CFR on such games
//! converges to the same weaker equilibrium notion the engine already accepts for raked games
//! (an approximate equilibrium of the non-zero-sum game rather than an exact Nash equilibrium
//! of a zero-sum one), and ICM curvature is typically larger than rake. The exploitability
//! reported by [`compute_exploitability`] — routed through the same current-EV-subtracting
//! branch as raked games, and measured in $ — remains the honest quality signal and should be
//! surfaced, not hidden.
//!
//! [`PostFlopGame::set_icm_effect`]: crate::PostFlopGame::set_icm_effect
//! [`compute_exploitability`]: crate::compute_exploitability

/// The tournament situation an ICM solve maps chip outcomes into.
///
/// Deliberately **not** serializable: the file format records nothing about ICM, so a stored
/// solution reopened without re-applying the effect silently reads back in chip space.
#[derive(Debug, Clone, PartialEq)]
pub struct IcmConfig {
    /// Remaining prizes, non-increasing, in any $ unit. Between 1 and `stacks.len()` entries;
    /// places beyond the list pay nothing.
    pub payouts: Vec<f64>,
    /// Every remaining player's stack **behind** at the start of the solved street, in the
    /// caller's chip units. The two contestants' stacks exclude their share of the starting
    /// pot. Between 2 and 10 entries, all positive.
    pub stacks: Vec<f64>,
    /// Index into `stacks` of the OOP player (player 0).
    pub oop_seat: usize,
    /// Index into `stacks` of the IP player (player 1).
    pub ip_seat: usize,
}

impl IcmConfig {
    /// Checks the configuration against the tree it is about to be applied to.
    ///
    /// The one cross-check that matters is units consistency: the shorter contestant stack must
    /// equal the tree's effective stack, so that the tree's all-in terminals correspond to a
    /// real bust. Everything else is internal (stack positivity, payout monotonicity, seat
    /// sanity, and a refusal of flat payouts, which leave nothing to solve).
    pub fn validate(&self, starting_pot: i32, effective_stack: i32) -> Result<(), String> {
        if starting_pot <= 0 {
            return Err(format!(
                "starting pot must be positive: starting_pot = {starting_pot}"
            ));
        }

        let num_players = self.stacks.len();
        if !(2..=10).contains(&num_players) {
            return Err(format!(
                "`stacks` must list between 2 and 10 players; got {num_players}"
            ));
        }

        for (i, &stack) in self.stacks.iter().enumerate() {
            if !stack.is_finite() {
                return Err(format!("`stacks[{i}]` must be a finite number"));
            }
            if stack <= 0.0 {
                return Err(format!("`stacks[{i}]` must be positive"));
            }
        }

        if self.payouts.is_empty() || self.payouts.len() > num_players {
            return Err(format!(
                "`payouts` must list between 1 and {num_players} prizes; got {}",
                self.payouts.len()
            ));
        }

        for (i, &payout) in self.payouts.iter().enumerate() {
            if !payout.is_finite() {
                return Err(format!("`payouts[{i}]` must be a finite number"));
            }
            if payout < 0.0 {
                return Err(format!("`payouts[{i}]` must not be negative"));
            }
        }

        if self.payouts.windows(2).any(|w| w[0] < w[1]) {
            return Err("`payouts` must be non-increasing".to_string());
        }

        // Places beyond the payout list implicitly pay 0.
        let last_payout = if self.payouts.len() < num_players {
            0.0
        } else {
            *self.payouts.last().unwrap()
        };
        if self.payouts[0] <= last_payout {
            return Err(
                "`payouts` are flat; every outcome pays the same and there is nothing to solve"
                    .to_string(),
            );
        }

        if self.oop_seat >= num_players {
            return Err(format!(
                "`oop_seat` must index into `stacks`: oop_seat = {}, players = {num_players}",
                self.oop_seat
            ));
        }

        if self.ip_seat >= num_players {
            return Err(format!(
                "`ip_seat` must index into `stacks`: ip_seat = {}, players = {num_players}",
                self.ip_seat
            ));
        }

        if self.oop_seat == self.ip_seat {
            return Err("`oop_seat` and `ip_seat` must name different players".to_string());
        }

        let min_contestant = self.stacks[self.oop_seat].min(self.stacks[self.ip_seat]);
        if min_contestant != f64::from(effective_stack) {
            return Err(format!(
                "`stacks` disagree with the tree: the shorter contestant stack is \
                 {min_contestant} where the effective stack is {effective_stack}; the tree's \
                 all-ins would not correspond to a real bust"
            ));
        }

        Ok(())
    }
}

/// Computes the Malmuth–Harville $ equity of every seat.
///
/// The model: the winner is drawn with probability proportional to stacks; the runner-up is
/// drawn from the remainder with probability proportional to stacks; and so on. A seat with a
/// zero stack has probability 0 for every place contested by a positive stack; once only
/// zero-stack seats remain, they split the remaining places uniformly (with at most one busted
/// player — the only case a terminal of this engine can produce — this is deterministic).
/// Places beyond `payouts.len()` pay nothing.
///
/// Implemented as a dynamic program over subsets of placed players — `f[mask]` is the
/// probability that exactly the players in `mask` occupy the top `mask.count_ones()` places —
/// rather than the naive permutation sum: *O*(2^N · N) against
/// *O*(N! / (N − #payouts)!), which is what makes applying the effect to 9- and 10-handed
/// configurations instant instead of taking seconds per terminal amount.
pub fn icm_equity(payouts: &[f64], stacks: &[f64]) -> Vec<f64> {
    let num_players = stacks.len();
    let num_paid = payouts.len().min(num_players);
    let mut equity = vec![0.0; num_players];

    if num_paid == 0 {
        return equity;
    }

    // f[mask] is only ever needed for masks of size < num_paid: a mask of size k hands out
    // place k + 1, and places beyond num_paid contribute nothing.
    let mut f = vec![0.0_f64; 1 << num_players];
    f[0] = 1.0;

    for mask in 0..(1_usize << num_players) {
        let placed = (mask as u32).count_ones() as usize;
        if placed >= num_paid {
            continue;
        }

        let prob = f[mask];
        if prob == 0.0 {
            continue;
        }

        let mut remaining_total = 0.0;
        let mut remaining_count = 0.0;
        for (i, &stack) in stacks.iter().enumerate() {
            if mask & (1 << i) == 0 {
                remaining_total += stack;
                remaining_count += 1.0;
            }
        }

        for (i, &stack) in stacks.iter().enumerate() {
            if mask & (1 << i) != 0 {
                continue;
            }
            // Proportional to stacks while any positive stack remains; uniform among the
            // zero-stack leftovers once none does.
            let p = if remaining_total > 0.0 {
                prob * stack / remaining_total
            } else {
                prob / remaining_count
            };
            if p > 0.0 {
                equity[i] += payouts[placed] * p;
                if placed + 1 < num_paid {
                    f[mask | (1 << i)] += p;
                }
            }
        }
    }

    equity
}

/// The ICM analog of the starting pot: what the contested chips are worth in $.
///
/// Defined as `(Δ_oop + Δ_ip) / 2` where `Δ_i` is the $ swing for contestant `i` between
/// winning and losing the starting pot with no further betting. It reduces to exactly
/// `starting_pot` when payouts are linear in chips (heads up with the prize gap equal to the
/// chips in play), and it is the value exploitability targets should scale against — "0.5% of
/// the pot" in an ICM solve means 0.5% of this.
pub fn icm_pot_value(icm: &IcmConfig, starting_pot: i32) -> f64 {
    let pot = f64::from(starting_pot);

    let equity_when = |winner: usize| {
        let mut stacks = icm.stacks.clone();
        stacks[winner] += pot;
        icm_equity(&icm.payouts, &stacks)
    };

    let oop_wins = equity_when(icm.oop_seat);
    let ip_wins = equity_when(icm.ip_seat);

    let delta_oop = oop_wins[icm.oop_seat] - ip_wins[icm.oop_seat];
    let delta_ip = ip_wins[icm.ip_seat] - oop_wins[icm.ip_seat];

    0.5 * (delta_oop + delta_ip)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malmuth_harville_matches_a_hand_computed_three_player_example() {
        // Stacks 50 / 30 / 20, payouts 100 / 60. By hand:
        // P(A 1st) = 0.5; P(A 2nd) = 0.3 * 50/70 + 0.2 * 50/80 = 0.33928571...
        let payouts = [100.0, 60.0];
        let stacks = [50.0, 30.0, 20.0];
        let equity = icm_equity(&payouts, &stacks);

        let p_a_2nd = 0.3 * 50.0 / 70.0 + 0.2 * 50.0 / 80.0;
        assert!((equity[0] - (0.5 * 100.0 + p_a_2nd * 60.0)).abs() < 1e-12);

        let p_b_2nd = 0.5 * 30.0 / 50.0 + 0.2 * 30.0 / 80.0;
        assert!((equity[1] - (0.3 * 100.0 + p_b_2nd * 60.0)).abs() < 1e-12);

        let p_c_2nd = 0.5 * 20.0 / 50.0 + 0.3 * 20.0 / 70.0;
        assert!((equity[2] - (0.2 * 100.0 + p_c_2nd * 60.0)).abs() < 1e-12);

        // The prize pool is conserved.
        let pool: f64 = equity.iter().sum();
        assert!((pool - 160.0).abs() < 1e-12);
    }

    #[test]
    fn payouts_shorter_than_the_player_list_pay_the_tail_nothing() {
        // One prize, three players: equity is the win probability times the prize.
        let equity = icm_equity(&[90.0], &[60.0, 30.0, 10.0]);
        assert!((equity[0] - 54.0).abs() < 1e-12);
        assert!((equity[1] - 27.0).abs() < 1e-12);
        assert!((equity[2] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn a_zero_stack_gets_nothing_while_positive_stacks_remain() {
        // The busted player is guaranteed last: they collect only the unpaid place.
        let equity = icm_equity(&[100.0, 60.0], &[70.0, 30.0, 0.0]);
        assert!((equity[2] - 0.0).abs() < 1e-12);
        let pool: f64 = equity.iter().sum();
        assert!((pool - 160.0).abs() < 1e-12);

        // With as many payouts as players, the busted player collects exactly the last prize.
        let equity = icm_equity(&[100.0, 60.0, 40.0], &[70.0, 30.0, 0.0]);
        assert!((equity[2] - 40.0).abs() < 1e-12);
    }

    #[test]
    fn heads_up_equity_is_linear_in_chips() {
        let payouts = [100.0, 60.0];
        for (s0, s1) in [(500.0, 500.0), (900.0, 100.0), (250.0, 750.0)] {
            let equity = icm_equity(&payouts, &[s0, s1]);
            let expected0 = 60.0 + 40.0 * s0 / (s0 + s1);
            assert!((equity[0] - expected0).abs() < 1e-12);
            assert!((equity[0] + equity[1] - 160.0).abs() < 1e-12);
        }
    }

    #[test]
    fn pot_value_reduces_to_the_starting_pot_under_chip_linear_payouts() {
        // Heads up, winner takes a prize equal to every chip in play (stacks + pot), loser
        // takes nothing: a chip is worth exactly a chip, so the pot is worth the pot.
        let icm = IcmConfig {
            payouts: vec![500.0 + 300.0 + 200.0, 0.0],
            stacks: vec![500.0, 300.0],
            oop_seat: 0,
            ip_seat: 1,
        };
        let value = icm_pot_value(&icm, 200);
        assert!((value - 200.0).abs() < 1e-9, "{value}");
    }

    #[test]
    fn validation_names_the_offending_field() {
        let base = || IcmConfig {
            payouts: vec![50.0, 30.0, 20.0],
            stacks: vec![900.0, 1400.0, 2500.0],
            oop_seat: 0,
            ip_seat: 1,
        };
        assert!(base().validate(200, 900).is_ok());

        type Mutation = Box<dyn Fn(&mut IcmConfig)>;
        let cases: Vec<(Mutation, &str)> = vec![
            (Box::new(|c| c.stacks = vec![900.0]), "`stacks`"),
            (Box::new(|c| c.stacks[2] = 0.0), "`stacks[2]`"),
            (Box::new(|c| c.stacks[1] = f64::NAN), "`stacks[1]`"),
            (Box::new(|c| c.payouts = vec![]), "`payouts`"),
            (
                Box::new(|c| c.payouts = vec![50.0, 30.0, 20.0, 10.0]),
                "`payouts`",
            ),
            (
                Box::new(|c| c.payouts = vec![50.0, 60.0, 20.0]),
                "non-increasing",
            ),
            (Box::new(|c| c.payouts = vec![50.0, 50.0, 50.0]), "flat"),
            (Box::new(|c| c.payouts[1] = -1.0), "`payouts[1]`"),
            (Box::new(|c| c.oop_seat = 3), "`oop_seat`"),
            (Box::new(|c| c.ip_seat = 0), "different players"),
            (Box::new(|c| c.stacks[0] = 901.0), "effective stack"),
        ];

        for (mutate, needle) in cases {
            let mut config = base();
            mutate(&mut config);
            let err = config.validate(200, 900).unwrap_err();
            assert!(err.contains(needle), "`{err}` does not mention {needle}");
        }
    }

    #[test]
    fn flat_payouts_shorter_than_the_table_are_not_flat() {
        // Two equal prizes over three players: the implicit 0 for third place makes the
        // payouts non-flat, so there is something to solve.
        let icm = IcmConfig {
            payouts: vec![50.0, 50.0],
            stacks: vec![900.0, 1400.0, 2500.0],
            oop_seat: 0,
            ip_seat: 1,
        };
        assert!(icm.validate(200, 900).is_ok());
    }
}
