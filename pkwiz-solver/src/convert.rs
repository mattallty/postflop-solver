//! The boundary where `postflop_solver::Range` starts and stops existing.
//!
//! `postflop_solver::Range` exists in this module and nowhere else. Everything above it speaks
//! [`pkwiz_range::Range`] and [`pkwiz_range::Card`], which is what keeps the engine's types — and
//! so its licence — from reaching anything that merely drives this binary.
//!
//! The card encodings are luckily identical (`rank * 4 + suit`, clubs `0` through spades `3`,
//! deuce `0` through ace `12`), so a card conversion is a cast and not a lookup table. That is an
//! agreement worth asserting rather than assuming, and [`tests`] does.

use pkwiz_range::Card as PkCard;
use pkwiz_range::Range as PkRange;
use postflop_solver::{Card as EngineCard, Range as EngineRange};

/// Our range, as the engine wants it.
///
/// The engine stores a weight per 1326 combinations; ours stores only the combinations that are
/// in the range. [`EngineRange::from_hands_weights`] is the public door between the two, so this
/// never touches the engine's internal indexing.
///
/// # Errors
///
/// Only if a weight is outside `0.0..=1.0`, which [`PkRange`] already prevents — it is checked
/// again here because the alternative is trusting an invariant across a crate boundary.
pub fn to_engine_range(range: &PkRange) -> Result<EngineRange, String> {
    let mut hands = Vec::with_capacity(range.len());
    let mut weights = Vec::with_capacity(range.len());
    for wc in range.combos() {
        let [a, b] = wc.combo.cards();
        hands.push((to_engine_card(a), to_engine_card(b)));
        weights.push(wc.weight as f32);
    }
    EngineRange::from_hands_weights(&hands, &weights)
}

/// One card, our encoding to theirs.
#[inline]
#[must_use]
pub const fn to_engine_card(card: PkCard) -> EngineCard {
    card.index()
}

/// One card, theirs to ours.
///
/// # Errors
///
/// If the engine hands back something outside `0..52`, which would mean `NOT_DEALT` leaked out of
/// a place it should not have.
pub fn from_engine_card(card: EngineCard) -> Result<PkCard, String> {
    PkCard::from_index(card)
        .ok_or_else(|| format!("engine returned card id {card}, which is not a card"))
}

/// A pair of hole cards, theirs to ours, rendered the way a player writes it.
///
/// # Errors
///
/// As [`from_engine_card`], or if the two cards are the same.
pub fn hole_to_string(hole: (EngineCard, EngineCard)) -> Result<String, String> {
    if hole.0 == hole.1 {
        return Err(format!(
            "hole cards are the same card (id {}), which no deck deals",
            hole.0
        ));
    }
    let (a, b) = (from_engine_card(hole.0)?, from_engine_card(hole.1)?);
    // Higher card first, which is how both the engine and our own `Combo` render a hand.
    let (hi, lo) = if a.index() >= b.index() {
        (a, b)
    } else {
        (b, a)
    };
    Ok(format!("{hi}{lo}"))
}

/// One engine action, rendered the way `NodeView.actions` shows it.
///
/// This and [`action_from_str`] are deliberate inverses: hosts compose edit lines by copying
/// strings out of `NodeView`, so the grammar is exact-match on this rendering.
#[must_use]
pub fn action_to_string(action: &postflop_solver::Action) -> String {
    use postflop_solver::Action;
    match action {
        Action::None => "None".to_owned(),
        Action::Fold => "Fold".to_owned(),
        Action::Check => "Check".to_owned(),
        Action::Call => "Call".to_owned(),
        Action::Bet(n) => format!("Bet({n})"),
        Action::Raise(n) => format!("Raise({n})"),
        Action::AllIn(n) => format!("AllIn({n})"),
        Action::Chance(c) => {
            from_engine_card(*c).map_or_else(|_| "Chance(?)".to_owned(), |c| format!("Chance({c})"))
        }
    }
}

/// The inverse of [`action_to_string`], for the actions a line may contain.
///
/// Accepts exactly `Fold`, `Check`, `Call`, and `Bet(n)`/`Raise(n)`/`AllIn(n)` with `n` a
/// positive integer — case-sensitive, because exactness against the `NodeView` rendering is
/// the contract, and a second, lenient grammar would have to stay compatible forever.
///
/// # Errors
///
/// A reason naming what was wrong. `Chance(..)` gets a targeted one: chance actions are
/// omitted from lines, because the tree treats all turn/river cards as one node.
pub fn action_from_str(s: &str) -> Result<postflop_solver::Action, String> {
    use postflop_solver::Action;
    match s {
        "Fold" => return Ok(Action::Fold),
        "Check" => return Ok(Action::Check),
        "Call" => return Ok(Action::Call),
        _ => {}
    }
    if s.starts_with("Chance(") {
        return Err(
            "chance actions are omitted from lines; the tree treats all turn/river \
                    cards as one node, so card-specific edits are not supported"
                .to_owned(),
        );
    }
    for (prefix, build) in [
        ("Bet(", Action::Bet as fn(i32) -> Action),
        ("Raise(", Action::Raise as fn(i32) -> Action),
        ("AllIn(", Action::AllIn as fn(i32) -> Action),
    ] {
        if let Some(rest) = s.strip_prefix(prefix) {
            let Some(digits) = rest.strip_suffix(')') else {
                return Err(format!("`{s}` is missing its closing parenthesis"));
            };
            let amount = digits
                .parse::<i64>()
                .map_err(|_| format!("`{digits}` is not an amount"))?;
            if amount <= 0 || amount > i64::from(i32::MAX) {
                return Err(format!("`{digits}` is not a positive 32-bit amount"));
            }
            return Ok(build(amount as i32));
        }
    }
    Err(format!(
        "`{s}` is not an action; expected Fold, Check, Call, Bet(n), Raise(n) or AllIn(n)"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actions_round_trip_between_render_and_parse() {
        use postflop_solver::Action;
        for action in [
            Action::Fold,
            Action::Check,
            Action::Call,
            Action::Bet(1),
            Action::Raise(188),
            Action::AllIn(i32::MAX),
        ] {
            let rendered = action_to_string(&action);
            assert_eq!(action_from_str(&rendered), Ok(action), "{rendered}");
        }
        for bad in [
            "",
            "bet(50)",
            "Bet(0)",
            "Bet(-5)",
            "Bet(2147483648)",
            "Bet(50",
            "None",
        ] {
            assert!(action_from_str(bad).is_err(), "`{bad}` should not parse");
        }
        assert!(action_from_str("Chance(Qc)")
            .unwrap_err()
            .contains("chance actions are omitted"));
    }

    #[test]
    fn the_two_card_encodings_agree_on_all_fifty_two() {
        // If this ever fails, `to_engine_card` has to become a table and every board and range in
        // a saved solution is suspect. Cheap insurance against a silent renumbering upstream.
        for i in 0..52u8 {
            let ours = PkCard::from_index(i).unwrap();
            let theirs = to_engine_card(ours);
            assert_eq!(theirs, i);
            assert_eq!(
                postflop_solver::card_to_string(theirs).unwrap(),
                ours.to_string(),
                "card {i} renders differently on the two sides"
            );
            assert_eq!(from_engine_card(theirs).unwrap(), ours);
        }
        assert!(from_engine_card(52).is_err());
    }

    #[test]
    fn a_range_crosses_the_boundary_with_its_weights_intact() {
        let ours = PkRange::parse("QQ+, AKs:0.5").unwrap();
        let theirs = to_engine_range(&ours).unwrap();

        // 18 pairs at full weight plus 4 suited combos at a half.
        let total: f32 = theirs.raw_data().iter().sum();
        assert!((total - 20.0).abs() < 1e-5, "{total}");
        assert_eq!(theirs.raw_data().iter().filter(|w| **w > 0.0).count(), 22);

        // And the weights landed on the right combinations, not merely on the right count.
        let ace = 12;
        let king = 11;
        let queen = 10;
        assert_eq!(theirs.get_weight_pair(queen), 1.0);
        assert_eq!(theirs.get_weight_suited(ace, king), 0.5);
        assert_eq!(theirs.get_weight_offsuit(ace, king), 0.0);
        assert_eq!(theirs.get_weight_pair(9), 0.0);
    }

    #[test]
    fn an_explicit_combination_survives_suit_for_suit() {
        let ours = PkRange::parse("AhKd").unwrap();
        let theirs = to_engine_range(&ours).unwrap();
        assert_eq!(theirs.raw_data().iter().filter(|w| **w > 0.0).count(), 1);
        let (hands, weights) = theirs.get_hands_weights(0);
        assert_eq!(hands.len(), 1);
        assert_eq!(weights[0], 1.0);
        assert_eq!(hole_to_string(hands[0]).unwrap(), "AhKd");
    }

    #[test]
    fn every_combination_round_trips() {
        let ours = PkRange::any();
        let theirs = to_engine_range(&ours).unwrap();
        assert_eq!(theirs, EngineRange::ones());

        let (hands, _) = theirs.get_hands_weights(0);
        assert_eq!(hands.len(), 1326);
        let mut rendered: Vec<String> = hands.iter().map(|h| hole_to_string(*h).unwrap()).collect();
        let mut mine: Vec<String> = ours.combos().iter().map(|c| c.combo.to_string()).collect();
        rendered.sort();
        mine.sort();
        assert_eq!(rendered, mine);
    }

    #[test]
    fn an_empty_range_is_empty_rather_than_full() {
        let theirs = to_engine_range(&PkRange::parse("").unwrap()).unwrap();
        assert!(theirs.is_empty());
    }
}
