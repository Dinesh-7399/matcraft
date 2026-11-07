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

use std::ops::{Add, Div, Mul, Neg, Sub};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum AutodiffError {
    #[error("Dimension mismatch in Jacobian evaluation: expected {expected} outputs, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Invalid variable index {0} in reverse-mode tape")]
    InvalidTapeIndex(usize),
}

/// Forward-mode Automatic Differentiation Dual Number (`val + eps * der`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Dual64 {
    pub val: f64,
    pub der: f64,
}

impl Dual64 {
    #[must_use]
    pub const fn constant(val: f64) -> Self {
        Self { val, der: 0.0 }
    }

    #[must_use]
    pub const fn variable(val: f64) -> Self {
        Self { val, der: 1.0 }
    }

    #[must_use]
    pub const fn new(val: f64, der: f64) -> Self {
        Self { val, der }
    }

    #[must_use]
    pub fn tanh(self) -> Self {
        let t = self.val.tanh();
        Self {
            val: t,
            der: self.der * (1.0 - t * t),
        }
    }

    pub fn sin(self) -> Self {
        let (s, c) = self.val.sin_cos();
        Self::new(s, self.der * c)
    }

    #[must_use]
    pub fn cos(self) -> Self {
        let (s, c) = self.val.sin_cos();
        Self::new(c, -self.der * s)
    }

    #[must_use]
    pub fn exp(self) -> Self {
        let e = self.val.exp();
        Self::new(e, self.der * e)
    }

    #[must_use]
    pub fn ln(self) -> Self {
        Self::new(self.val.ln(), self.der / self.val)
    }

    #[must_use]
    pub fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        Self::new(s, self.der / (2.0 * s))
    }

    #[must_use]
    pub fn powi(self, n: i32) -> Self {
        Self::new(
            self.val.powi(n),
            self.der * f64::from(n) * self.val.powi(n.saturating_sub(1)),
        )
    }
}

impl Add for Dual64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.val + rhs.val, self.der + rhs.der)
    }
}

impl Sub for Dual64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.val - rhs.val, self.der - rhs.der)
    }
}

impl Mul for Dual64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.val * rhs.val,
            self.der * rhs.val + self.val * rhs.der,
        )
    }
}

impl Div for Dual64 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.val * rhs.val;
        Self::new(
            self.val / rhs.val,
            (self.der * rhs.val - self.val * rhs.der) / denom,
        )
    }
}

impl Neg for Dual64 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.val, -self.der)
    }
}

/// Computes the exact value $F(x) \in \mathbb{R}^m$ and analytical Jacobian $J(x) \in \mathbb{R}^{m \times n}$
/// (column-major) using forward-mode dual numbers—zero finite-difference approximation error!
pub fn jacobian_forward<F>(x: &[f64], f: F) -> Result<(Vec<f64>, Vec<f64>), AutodiffError>
where
    F: Fn(&[Dual64]) -> Vec<Dual64>,
{
    let n = x.len();
    let mut dual_in: Vec<Dual64> = x.iter().map(|&v| Dual64::constant(v)).collect();
    if n == 0 {
        let out = f(&dual_in);
        let vals = out.iter().map(|d| d.val).collect();
        return Ok((vals, Vec::new()));
    }

    let mut fx = Vec::new();
    let mut jac_col_major = Vec::new();
    let mut m_expected: Option<usize> = None;

    for col in 0..n {
        if let Some(slot) = dual_in.get_mut(col) {
            slot.der = 1.0;
        }
        let out = f(&dual_in);
        if let Some(m) = m_expected {
            if out.len() != m {
                return Err(AutodiffError::DimensionMismatch {
                    expected: m,
                    actual: out.len(),
                });
            }
        } else {
            m_expected = Some(out.len());
            fx = out.iter().map(|d| d.val).collect();
            jac_col_major.reserve(out.len().saturating_mul(n));
        }

        for d in &out {
            jac_col_major.push(d.der);
        }

        if let Some(slot) = dual_in.get_mut(col) {
            slot.der = 0.0;
        }
    }

    Ok((fx, jac_col_major))
}

#[derive(Clone, Copy, Debug)]
struct TapeNode {
    value: f64,
    parents: [Option<usize>; 2],
    partials: [f64; 2],
}

/// Reverse-mode Wengert Tape for $O(1)$-pass gradients of scalar objectives $L: \mathbb{R}^n \to \mathbb{R}$.
#[derive(Clone, Debug, Default)]
pub struct ReverseTape {
    nodes: Vec<TapeNode>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TapeVar(pub usize);

impl ReverseTape {
    #[must_use]
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn var(&mut self, value: f64) -> TapeVar {
        let idx = self.nodes.len();
        self.nodes.push(TapeNode {
            value,
            parents: [None, None],
            partials: [0.0, 0.0],
        });
        TapeVar(idx)
    }

    pub fn value(&self, v: TapeVar) -> Result<f64, AutodiffError> {
        self.nodes
            .get(v.0)
            .map(|n| n.value)
            .ok_or(AutodiffError::InvalidTapeIndex(v.0))
    }

    pub fn add(&mut self, a: TapeVar, b: TapeVar) -> Result<TapeVar, AutodiffError> {
        let va = self.value(a)?;
        let vb = self.value(b)?;
        let idx = self.nodes.len();
        self.nodes.push(TapeNode {
            value: va + vb,
            parents: [Some(a.0), Some(b.0)],
            partials: [1.0, 1.0],
        });
        Ok(TapeVar(idx))
    }

    pub fn sub(&mut self, a: TapeVar, b: TapeVar) -> Result<TapeVar, AutodiffError> {
        let va = self.value(a)?;
        let vb = self.value(b)?;
        let idx = self.nodes.len();
        self.nodes.push(TapeNode {
            value: va - vb,
            parents: [Some(a.0), Some(b.0)],
            partials: [1.0, -1.0],
        });
        Ok(TapeVar(idx))
    }

    pub fn mul(&mut self, a: TapeVar, b: TapeVar) -> Result<TapeVar, AutodiffError> {
        let va = self.value(a)?;
        let vb = self.value(b)?;
        let idx = self.nodes.len();
        self.nodes.push(TapeNode {
            value: va * vb,
            parents: [Some(a.0), Some(b.0)],
            partials: [vb, va],
        });
        Ok(TapeVar(idx))
    }

    pub fn sin(&mut self, a: TapeVar) -> Result<TapeVar, AutodiffError> {
        let va = self.value(a)?;
        let (s, c) = va.sin_cos();
        let idx = self.nodes.len();
        self.nodes.push(TapeNode {
            value: s,
            parents: [Some(a.0), None],
            partials: [c, 0.0],
        });
        Ok(TapeVar(idx))
    }

    /// Computes $\frac{\partial \text{output}}{\partial v_i}$ for all variables on the tape.
    pub fn backward(&self, output: TapeVar) -> Result<Vec<f64>, AutodiffError> {
        if output.0 >= self.nodes.len() {
            return Err(AutodiffError::InvalidTapeIndex(output.0));
        }
        let mut adjoints = vec![0.0; self.nodes.len()];
        if let Some(out_adj) = adjoints.get_mut(output.0) {
            *out_adj = 1.0;
        }
        for i in (0..=output.0).rev() {
            let adj = adjoints.get(i).copied().unwrap_or(0.0);
            if adj == 0.0 {
                continue;
            }
            if let Some(node) = self.nodes.get(i) {
                for (p_opt, &w) in node.parents.iter().zip(&node.partials) {
                    if let Some(p_idx) = *p_opt {
                        if let Some(parent_adj) = adjoints.get_mut(p_idx) {
                            *parent_adj += adj * w;
                        }
                    }
                }
            }
        }
        Ok(adjoints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_jacobian_and_reverse_tape() -> Result<(), Box<dyn std::error::Error>> {
        // f(x, y) = [x^2 * y, sin(x) + y] at (2.0, 3.0)
        // J = [[2xy, x^2], [cos(x), 1]] = [[12, 4], [cos(2), 1]]
        let (fx, jac) = jacobian_forward(&[2.0, 3.0], |vars| {
            let x = vars.first().copied().unwrap_or_default();
            let y = vars.get(1).copied().unwrap_or_default();
            vec![x.powi(2) * y, x.sin() + y]
        })?;
        assert!((fx.first().copied().unwrap_or(0.0) - 12.0).abs() < 1e-12);
        assert!((jac.first().copied().unwrap_or(0.0) - 12.0).abs() < 1e-12);
        assert!((jac.get(2).copied().unwrap_or(0.0) - 4.0).abs() < 1e-12);

        // Reverse tape: L(x, y) = x * y + sin(x) at (2.0, 3.0)
        let mut tape = ReverseTape::new();
        let x = tape.var(2.0);
        let y = tape.var(3.0);
        let xy = tape.mul(x, y)?;
        let sx = tape.sin(x)?;
        let loss = tape.add(xy, sx)?;
        let grads = tape.backward(loss)?;
        assert!((grads.first().copied().unwrap_or(0.0) - (3.0 + 2.0_f64.cos())).abs() < 1e-12);
        assert!((grads.get(1).copied().unwrap_or(0.0) - 2.0).abs() < 1e-12);
        Ok(())
    }
}
