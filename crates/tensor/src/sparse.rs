use crate::{Tensor, TensorError};
use num_core::{MemoryBudget, SiUnit};
use std::sync::Arc;

/// Compressed Sparse Column (CSC) matrix — matches MATLAB's native sparse layout.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseCsc<T: Clone = f64> {
    rows: usize,
    cols: usize,
    col_ptrs: Vec<usize>,
    row_indices: Vec<usize>,
    values: Vec<T>,
    unit: SiUnit,
}

impl SparseCsc<f64> {
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        triplets: &[(usize, usize, f64)],
        unit: SiUnit,
        budget: &Arc<MemoryBudget>,
    ) -> Result<Self, TensorError> {
        let _res = budget.try_reserve_elements(
            cols.saturating_add(1).saturating_add(triplets.len() * 2),
            std::mem::size_of::<f64>(),
        )?;
        for &(r, c, _) in triplets {
            if r >= rows || c >= cols {
                return Err(TensorError::IndexOutOfBounds {
                    index: vec![r, c],
                    shape: vec![rows, cols],
                });
            }
        }
        let mut sorted = triplets.to_vec();
        sorted.sort_by_key(|&(r, c, _)| (c, r));

        let mut col_ptrs = vec![0usize; cols.saturating_add(1)];
        let mut row_indices = Vec::with_capacity(sorted.len());
        let mut values = Vec::with_capacity(sorted.len());

        let mut last_coord: Option<(usize, usize)> = None;
        for (r, c, v) in sorted {
            if v == 0.0 {
                continue;
            }
            if last_coord == Some((r, c)) {
                if let Some(last_v) = values.last_mut() {
                    *last_v += v;
                }
            } else {
                row_indices.push(r);
                values.push(v);
                if let Some(slot) = col_ptrs.get_mut(c.saturating_add(1)) {
                    *slot = slot.saturating_add(1);
                }
                last_coord = Some((r, c));
            }
        }

        // Prefix sum for column pointers
        let mut running = 0usize;
        for ptr in &mut col_ptrs {
            running = running.saturating_add(*ptr);
            *ptr = running;
        }

        Ok(Self {
            rows,
            cols,
            col_ptrs,
            row_indices,
            values,
            unit,
        })
    }

    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparse Matrix-Vector Multiplication (`y = A * x`).
    pub fn spmv(&self, x: &[f64]) -> Result<Vec<f64>, TensorError> {
        if x.len() != self.cols {
            return Err(TensorError::ShapeMismatch {
                op: "spmv",
                lhs: vec![self.rows, self.cols],
                rhs: vec![x.len(), 1],
            });
        }
        let mut y = vec![0.0; self.rows];
        for c in 0..self.cols {
            let start = self.col_ptrs.get(c).copied().unwrap_or(0);
            let end = self.col_ptrs.get(c.saturating_add(1)).copied().unwrap_or(0);
            let xc = x.get(c).copied().unwrap_or(0.0);
            for idx in start..end {
                if let (Some(&r), Some(&val)) = (self.row_indices.get(idx), self.values.get(idx)) {
                    if let Some(slot) = y.get_mut(r) {
                        *slot += val * xc;
                    }
                }
            }
        }
        Ok(y)
    }

    pub fn to_dense(&self, budget: &Arc<MemoryBudget>) -> Result<Tensor<f64>, TensorError> {
        let mut t = Tensor::full_with_budget(self.rows, self.cols, 0.0, self.unit, budget)?;
        for c in 0..self.cols {
            let start = self.col_ptrs.get(c).copied().unwrap_or(0);
            let end = self.col_ptrs.get(c.saturating_add(1)).copied().unwrap_or(0);
            for idx in start..end {
                if let (Some(&r), Some(&val)) = (self.row_indices.get(idx), self.values.get(idx)) {
                    t.set(r, c, val)?;
                }
            }
        }
        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cow_tiled_tensor_materializes_only_mutated_tile(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::new(16 * 1024 * 1024); // Only 16 MB budget!
        // Create a 10,000 x 10,000 matrix (logically 800 MB if dense!).
        // In MATLAB, this would immediately exceed a 16 MB budget and fail.
        // In MatCraft, it uses `TileData::Uniform(0.0)` and succeeds in < 1 MB!
        let mut a = Tensor::zeros(10_000, 10_000, &budget)?;
        assert_eq!(a.materialized_tile_count(), 0);

        // Clone `a` for undo history (O(1) Arc clone)
        let snapshot = a.clone();

        // Mutate a single element at (500, 500)
        a.set(500, 500, 42.0)?;
        assert_eq!(a.get(500, 500)?, 42.0);
        assert_eq!(snapshot.get(500, 500)?, 0.0);

        // Prove that ONLY 1 single 64x64 tile (32 KB) was materialized out of 24,649 tiles!
        assert_eq!(a.materialized_tile_count(), 1);
        assert_eq!(snapshot.materialized_tile_count(), 0);
        Ok(())
    }

    #[test]
    fn test_panic_hunt_tensor_bounds_and_units() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        let mut m = Tensor::zeros(4, 4, &budget)?;
        assert!(m.get(4, 0).is_err());
        assert!(m.get_1based(0, 1).is_err());
        assert!(m.set(0, 4, 1.0).is_err());

        let mut dist = Tensor::ones(2, 2, &budget)?;
        dist.set_unit(SiUnit::METER);
        let mut time = Tensor::ones(2, 2, &budget)?;
        time.set_unit(SiUnit::SECOND);

        // Adding meters to seconds must return Err, never panic
        assert!(dist.checked_add(&time).is_err());
        // Dividing meters by seconds yields m/s!
        let vel = dist.elem_div(&time)?;
        assert_eq!(vel.unit().to_string(), "m/s");
        Ok(())
    }
}
