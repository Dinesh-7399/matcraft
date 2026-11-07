#![forbid(unsafe_code)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::todo,
    clippy::unimplemented,
    clippy::unreachable
)]

//! NumPy-style broadcasting for element-wise tensor operations.

use crate::{Tensor, TensorError};
use num_core::{MemoryBudget, SiUnit};
use std::sync::Arc;

/// Perform element-wise operation on two tensors with NumPy/MATLAB-style broadcasting.
///
/// Supported shape pairs (m,n) + (m,1), (m,n) + (1,n), (m,n) + (1,1),
/// (1,n) + (m,n), (m,1) + (m,n), and identical shapes.
/// Returns `TensorError::ShapeMismatch` for truly incompatible shapes.
pub fn broadcast_elementwise(
    a: &Tensor<f64>,
    b: &Tensor<f64>,
    op: impl Fn(f64, f64) -> f64,
    budget: &Arc<MemoryBudget>,
) -> Result<Tensor<f64>, TensorError> {
    let [ar, ac] = a.shape();
    let [br, bc] = b.shape();

    // Determine output shape via broadcast rules
    let out_rows = broadcast_dim(ar, br).ok_or_else(|| TensorError::ShapeMismatch {
        op: "broadcast_elementwise",
        lhs: vec![ar, ac],
        rhs: vec![br, bc],
    })?;
    let out_cols = broadcast_dim(ac, bc).ok_or_else(|| TensorError::ShapeMismatch {
        op: "broadcast_elementwise",
        lhs: vec![ar, ac],
        rhs: vec![br, bc],
    })?;

    // Build output column-major
    let mut out = Vec::with_capacity(out_rows.saturating_mul(out_cols));
    for c in 0..out_cols {
        for r in 0..out_rows {
            let ai = if ar == 1 { 0 } else { r };
            let aj = if ac == 1 { 0 } else { c };
            let bi = if br == 1 { 0 } else { r };
            let bj = if bc == 1 { 0 } else { c };
            let av = a.get(ai, aj)?;
            let bv = b.get(bi, bj)?;
            out.push(op(av, bv));
        }
    }

    // Combine units: for a generic op, we use dimensionless output;
    // the caller can re-unit the result if needed.
    // We pick a's unit as a reasonable default (matches add/sub semantics).
    let out_unit = if a.unit() == SiUnit::DIMENSIONLESS || b.unit() == SiUnit::DIMENSIONLESS {
        SiUnit::DIMENSIONLESS
    } else {
        a.unit()
    };

    Tensor::from_column_major(out_rows, out_cols, out, out_unit, budget)
}

/// Returns the broadcast output dimension for two dimension sizes.
/// Follows NumPy rules: same size is ok, or one of them is 1.
fn broadcast_dim(a: usize, b: usize) -> Option<usize> {
    if a == b {
        Some(a)
    } else if a == 1 {
        Some(b)
    } else if b == 1 {
        Some(a)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_core::MemoryBudget;

    #[test]
    fn test_broadcast_m_n_plus_m_1() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        // a: 2x3, b: 2x1
        let a = Tensor::from_column_major(
            2, 3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            SiUnit::DIMENSIONLESS,
            &budget,
        )?;
        let b = Tensor::from_column_major(
            2, 1,
            vec![10.0, 20.0],
            SiUnit::DIMENSIONLESS,
            &budget,
        )?;
        let result = broadcast_elementwise(&a, &b, |x, y| x + y, &budget)?;
        assert_eq!(result.shape(), [2, 3]);
        // Row 0: 1+10=11, 3+10=13, 5+10=15
        // Row 1: 2+20=22, 4+20=24, 6+20=26
        assert!((result.get(0, 0)? - 11.0).abs() < 1e-12);
        assert!((result.get(1, 2)? - 26.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_broadcast_m_n_plus_1_n() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        // a: 3x2, b: 1x2
        let a = Tensor::from_column_major(
            3, 2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            SiUnit::DIMENSIONLESS,
            &budget,
        )?;
        let b = Tensor::from_column_major(
            1, 2,
            vec![10.0, 100.0],
            SiUnit::DIMENSIONLESS,
            &budget,
        )?;
        let result = broadcast_elementwise(&a, &b, |x, y| x + y, &budget)?;
        assert_eq!(result.shape(), [3, 2]);
        assert!((result.get(0, 0)? - 11.0).abs() < 1e-12); // 1+10
        assert!((result.get(0, 1)? - 104.0).abs() < 1e-12); // 4+100
        Ok(())
    }

    #[test]
    fn test_broadcast_1_1_scalar() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        let a = Tensor::from_column_major(
            2, 3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            SiUnit::DIMENSIONLESS,
            &budget,
        )?;
        let b = Tensor::from_column_major(1, 1, vec![2.0], SiUnit::DIMENSIONLESS, &budget)?;
        let result = broadcast_elementwise(&a, &b, |x, y| x * y, &budget)?;
        assert_eq!(result.shape(), [2, 3]);
        assert!((result.get(1, 2)? - 12.0).abs() < 1e-12); // 6*2
        Ok(())
    }

    #[test]
    fn test_broadcast_incompatible_shape_returns_error() {
        let budget = MemoryBudget::default_budget();
        let a = Tensor::from_column_major(2, 3, vec![0.0; 6], SiUnit::DIMENSIONLESS, &budget).unwrap();
        let b = Tensor::from_column_major(3, 2, vec![0.0; 6], SiUnit::DIMENSIONLESS, &budget).unwrap();
        let result = broadcast_elementwise(&a, &b, |x, y| x + y, &budget);
        assert!(result.is_err());
    }
}
