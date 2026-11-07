use serde::{Deserialize, Serialize};

/// Structural Differential-Algebraic Equation (DAE) Index Analyzer & Pantelides Algorithm (`dae_index.rs`).
/// Analyzes the bipartite graph between equations and highest-order state derivatives to:
/// 1. Compute the maximum bipartite matching (assigning each equation to a solved variable),
/// 2. Detect algebraic constraint equations that lack highest-order derivatives (high-index DAEs),
/// 3. Identify the exact set of constraint equations that must be symbolically differentiated
///    (Pantelides index reduction) to lower a high-index DAE (such as a constrained rigid-body
///    pendulum $x^2 + y^2 - L^2 = 0$) into an Index-1 system!
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PantelidesReport {
    pub num_equations: usize,
    pub num_variables: usize,
    pub matched_equations: usize,
    pub is_structurally_regular_index1: bool,
    pub equations_to_differentiate: Vec<usize>,
}

#[must_use]
pub fn analyze_dae_structure(
    num_equations: usize,
    num_variables: usize,
    highest_deriv_incidence: &[Vec<usize>],
) -> PantelidesReport {
    let mut var_matched_to_eq: Vec<Option<usize>> = vec![None; num_variables];
    let mut eq_is_matched = vec![false; num_equations];
    let mut equations_to_differentiate = Vec::new();

    for eq_idx in 0..num_equations {
        let mut visited_vars = vec![false; num_variables];
        let mut visited_eqs = vec![false; num_equations];
        if augment_matching(
            eq_idx,
            highest_deriv_incidence,
            &mut var_matched_to_eq,
            &mut visited_vars,
            &mut visited_eqs,
        ) {
            if let Some(slot) = eq_is_matched.get_mut(eq_idx) {
                *slot = true;
            }
        } else {
            // Pantelides theorem: all equations visited in the failed alternating tree form a
            // Minimally Structurally Singular (MSS) subset and must be differentiated!
            for (e_i, &vis) in visited_eqs.iter().enumerate() {
                if vis && !equations_to_differentiate.contains(&e_i) {
                    equations_to_differentiate.push(e_i);
                }
            }
        }
    }

    let matched_equations = eq_is_matched.iter().filter(|&&b| b).count();
    PantelidesReport {
        num_equations,
        num_variables,
        matched_equations,
        is_structurally_regular_index1: matched_equations == num_equations
            && equations_to_differentiate.is_empty(),
        equations_to_differentiate,
    }
}

fn augment_matching(
    eq_idx: usize,
    incidence: &[Vec<usize>],
    var_matched_to_eq: &mut [Option<usize>],
    visited_vars: &mut [bool],
    visited_eqs: &mut [bool],
) -> bool {
    if let Some(slot) = visited_eqs.get_mut(eq_idx) {
        *slot = true;
    }
    let Some(adj_vars) = incidence.get(eq_idx) else {
        return false;
    };
    for &v in adj_vars {
        if visited_vars.get(v).copied().unwrap_or(true) {
            continue;
        }
        if let Some(v_slot) = visited_vars.get_mut(v) {
            *v_slot = true;
        }
        let current_owner = var_matched_to_eq.get(v).copied().flatten();
        let can_reassign = match current_owner {
            None => true,
            Some(other_eq) => augment_matching(
                other_eq,
                incidence,
                var_matched_to_eq,
                visited_vars,
                visited_eqs,
            ),
        };
        if can_reassign {
            if let Some(m_slot) = var_matched_to_eq.get_mut(v) {
                *m_slot = Some(eq_idx);
            }
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pantelides_detects_high_index_cartesian_pendulum_constraint() {
        // Cartesian pendulum DAE in highest-derivative variables [x'', y'', T]:
        // Eq 0: m*x'' + T*x = 0      -> depends on vars [0, 2]
        // Eq 1: m*y'' + T*y + mg = 0 -> depends on vars [1, 2]
        // Eq 2: x^2 + y^2 - L^2 = 0  -> position constraint! Contains NO highest derivatives: []
        let incidence = vec![vec![0, 2], vec![1, 2], vec![]];
        let report = analyze_dae_structure(3, 3, &incidence);
        assert!(!report.is_structurally_regular_index1);
        // Pantelides pinpoints Eq 2 (the algebraic holonomic constraint) for differentiation!
        assert_eq!(report.equations_to_differentiate, vec![2]);
    }
}
