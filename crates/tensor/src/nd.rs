use crate::TensorError;
use num_core::{AllocReservation, MemoryBudget, SiUnit};
use std::sync::Arc;

/// Arbitrary N-Dimensional Tensor (`NdTensor<T>`) with column-major (MATLAB-compatible) strides.
/// Supports `ndims`, `reshape`, `permute`, `squeeze`, `sub2ind`, `ind2sub`, and dimensional reductions.
#[derive(Clone, Debug)]
pub struct NdTensor<T: Clone = f64> {
    shape: Vec<usize>,
    strides: Vec<usize>,
    unit: SiUnit,
    budget: Arc<MemoryBudget>,
    values: Arc<Vec<T>>,
    _reservation: Arc<AllocReservation>,
}

impl<T: Clone + Copy + PartialEq> NdTensor<T> {
    pub fn full(
        shape: &[usize],
        fill_value: T,
        unit: SiUnit,
        budget: &Arc<MemoryBudget>,
    ) -> Result<Self, TensorError> {
        let count = MemoryBudget::checked_element_count(shape)?;
        let res = budget.try_reserve_elements(count, std::mem::size_of::<T>())?;
        let strides = compute_column_major_strides(shape);
        Ok(Self {
            shape: shape.to_vec(),
            strides,
            unit,
            budget: Arc::clone(budget),
            values: Arc::new(vec![fill_value; count]),
            _reservation: Arc::new(res),
        })
    }

    pub fn from_col_major(
        shape: &[usize],
        data: Vec<T>,
        unit: SiUnit,
        budget: &Arc<MemoryBudget>,
    ) -> Result<Self, TensorError> {
        let expected = MemoryBudget::checked_element_count(shape)?;
        if data.len() != expected {
            return Err(TensorError::DataLengthMismatch {
                data_len: data.len(),
                shape: shape.to_vec(),
                expected,
            });
        }
        let res = budget.try_reserve_elements(expected, std::mem::size_of::<T>())?;
        let strides = compute_column_major_strides(shape);
        Ok(Self {
            shape: shape.to_vec(),
            strides,
            unit,
            budget: Arc::clone(budget),
            values: Arc::new(data),
            _reservation: Arc::new(res),
        })
    }

    #[must_use]
    pub fn ndims(&self) -> usize {
        self.shape.len()
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub fn numel(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub const fn unit(&self) -> SiUnit {
        self.unit
    }

    /// Converts 0-based multi-dimensional subscript `coords` to a linear column-major index (`sub2ind`).
    pub fn sub2ind(&self, coords: &[usize]) -> Result<usize, TensorError> {
        if coords.len() != self.shape.len() {
            return Err(TensorError::IndexOutOfBounds {
                index: coords.to_vec(),
                shape: self.shape.clone(),
            });
        }
        let mut linear = 0usize;
        for ((&c, &dim), &st) in coords.iter().zip(&self.shape).zip(&self.strides) {
            if c >= dim {
                return Err(TensorError::IndexOutOfBounds {
                    index: coords.to_vec(),
                    shape: self.shape.clone(),
                });
            }
            linear = linear.saturating_add(c.saturating_mul(st));
        }
        Ok(linear)
    }

    /// Converts a 0-based linear column-major index into multi-dimensional coordinates (`ind2sub`).
    pub fn ind2sub(&self, mut linear_idx: usize) -> Result<Vec<usize>, TensorError> {
        if linear_idx >= self.numel() {
            return Err(TensorError::IndexOutOfBounds {
                index: vec![linear_idx],
                shape: self.shape.clone(),
            });
        }
        let mut coords = vec![0usize; self.shape.len()];
        for i in (0..self.shape.len()).rev() {
            let st = self.strides.get(i).copied().unwrap_or(1).max(1);
            if let Some(slot) = coords.get_mut(i) {
                *slot = linear_idx / st;
            }
            linear_idx %= st;
        }
        Ok(coords)
    }

    pub fn get(&self, coords: &[usize]) -> Result<T, TensorError> {
        let idx = self.sub2ind(coords)?;
        self.values
            .get(idx)
            .copied()
            .ok_or_else(|| TensorError::IndexOutOfBounds {
                index: coords.to_vec(),
                shape: self.shape.clone(),
            })
    }

    pub fn set(&mut self, coords: &[usize], val: T) -> Result<(), TensorError> {
        let idx = self.sub2ind(coords)?;
        if Arc::strong_count(&self.values) > 1 {
            let res = self
                .budget
                .try_reserve_elements(self.values.len(), std::mem::size_of::<T>())?;
            self._reservation = Arc::new(res);
        }
        let vec_mut = Arc::make_mut(&mut self.values);
        let slot = vec_mut
            .get_mut(idx)
            .ok_or_else(|| TensorError::IndexOutOfBounds {
                index: coords.to_vec(),
                shape: self.shape.clone(),
            })?;
        *slot = val;
        Ok(())
    }

    /// Zero-copy N-dimensional `reshape(&[usize])`.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, TensorError> {
        let count = MemoryBudget::checked_element_count(new_shape)?;
        if count != self.numel() {
            return Err(TensorError::DataLengthMismatch {
                data_len: self.numel(),
                shape: new_shape.to_vec(),
                expected: count,
            });
        }
        Ok(Self {
            shape: new_shape.to_vec(),
            strides: compute_column_major_strides(new_shape),
            unit: self.unit,
            budget: Arc::clone(&self.budget),
            values: Arc::clone(&self.values),
            _reservation: Arc::clone(&self._reservation),
        })
    }

    /// Removes singleton dimensions of length 1 (`squeeze(A)`).
    pub fn squeeze(&self) -> Result<Self, TensorError> {
        let mut squeezed: Vec<usize> = self.shape.iter().copied().filter(|&d| d != 1).collect();
        if squeezed.is_empty() {
            squeezed.push(1);
        }
        self.reshape(&squeezed)
    }

    /// Permutes N-D array dimensions (`permute(A, order)` where `order` is a 0-based permutation of `0..ndims`).
    pub fn permute(&self, order: &[usize]) -> Result<Self, TensorError> {
        let nd = self.ndims();
        if order.len() != nd {
            return Err(TensorError::ShapeMismatch {
                op: "permute",
                lhs: self.shape.clone(),
                rhs: order.to_vec(),
            });
        }
        let mut seen = vec![false; nd];
        let mut new_shape = Vec::with_capacity(nd);
        for &axis in order {
            if axis >= nd || seen.get(axis).copied().unwrap_or(true) {
                return Err(TensorError::ShapeMismatch {
                    op: "permute (invalid axis permutation)",
                    lhs: self.shape.clone(),
                    rhs: order.to_vec(),
                });
            }
            if let Some(s) = seen.get_mut(axis) {
                *s = true;
            }
            new_shape.push(self.shape.get(axis).copied().unwrap_or(0));
        }

        let mut out = Self::full(
            &new_shape,
            self.values.first().copied().ok_or_else(|| TensorError::IndexOutOfBounds {
                index: vec![0],
                shape: self.shape.clone(),
            })?,
            self.unit,
            &self.budget,
        )?;

        for lin in 0..self.numel() {
            let old_coords = self.ind2sub(lin)?;
            let mut new_coords = vec![0usize; nd];
            for (new_ax, &old_ax) in order.iter().enumerate() {
                if let Some(slot) = new_coords.get_mut(new_ax) {
                    *slot = old_coords.get(old_ax).copied().unwrap_or(0);
                }
            }
            let val = self.values.get(lin).copied().ok_or_else(|| {
                TensorError::IndexOutOfBounds {
                    index: vec![lin],
                    shape: self.shape.clone(),
                }
            })?;
            out.set(&new_coords, val)?;
        }
        Ok(out)
    }
}

impl NdTensor<f64> {
    /// Sums elements along 0-based dimension `dim` (`sum(A, dim)`), collapsing `shape[dim]` to `1`.
    pub fn sum_dim(&self, dim: usize) -> Result<Self, TensorError> {
        if dim >= self.ndims() {
            return Err(TensorError::IndexOutOfBounds {
                index: vec![dim],
                shape: self.shape.clone(),
            });
        }
        let mut out_shape = self.shape.clone();
        if let Some(slot) = out_shape.get_mut(dim) {
            *slot = 1;
        }
        let mut out = Self::full(&out_shape, 0.0, self.unit, &self.budget)?;
        for lin in 0..self.numel() {
            let mut coords = self.ind2sub(lin)?;
            let v = self.get(&coords)?;
            if let Some(slot) = coords.get_mut(dim) {
                *slot = 0;
            }
            let prev = out.get(&coords)?;
            out.set(&coords, prev + v)?;
        }
        Ok(out)
    }
}

fn compute_column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut running = 1usize;
    for &d in shape {
        strides.push(running);
        running = running.saturating_mul(d.max(1));
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nd_tensor_permute_squeeze_and_sum_dim() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        let data: Vec<f64> = (1..=24).map(f64::from).collect();
        let t = NdTensor::from_col_major(&[2, 3, 4], data, SiUnit::VOLT, &budget)?;
        assert_eq!(t.ndims(), 3);

        // Permute axes [2, 0, 1] -> shape [4, 2, 3]
        let p = t.permute(&[2, 0, 1])?;
        assert_eq!(p.shape(), &[4, 2, 3]);
        assert_eq!(t.get(&[1, 2, 3])?, p.get(&[3, 1, 2])?);

        // Sum along middle axis (dim=1) -> shape [2, 1, 4], then squeeze -> [2, 4]
        let s1 = t.sum_dim(1)?.squeeze()?;
        assert_eq!(s1.shape(), &[2, 4]);
        assert_eq!(s1.unit(), SiUnit::VOLT);
        Ok(())
    }
}
