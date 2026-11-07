use crate::{Tensor, TensorError};

/// Represents a strided index slice `start:step:stop` (0-based inclusive of `start`, exclusive of `stop`
/// or generated via `SliceRange::matlab_1based`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceRange {
    pub start: usize,
    pub step: isize,
    pub count: usize,
}

impl SliceRange {
    pub const fn all(len: usize) -> Self {
        Self {
            start: 0,
            step: 1,
            count: len,
        }
    }

    /// Constructs a 0-based `SliceRange` from 1-based inclusive MATLAB `start1:step:stop1`.
    pub fn matlab_1based(
        start1: usize,
        step: isize,
        stop1: usize,
        dim_len: usize,
    ) -> Result<Self, TensorError> {
        if step == 0 || start1 == 0 || stop1 == 0 || start1 > dim_len || stop1 > dim_len {
            return Err(TensorError::InvalidSliceRange {
                start: start1,
                step,
                stop: stop1,
                len: dim_len,
            });
        }
        let s0 = start1 - 1;
        let e0 = stop1 - 1;
        if step > 0 {
            if e0 < s0 {
                return Ok(Self {
                    start: s0,
                    step,
                    count: 0,
                });
            }
            let span = e0 - s0;
            let count = (span / (step as usize)).saturating_add(1);
            Ok(Self {
                start: s0,
                step,
                count,
            })
        } else {
            if s0 < e0 {
                return Ok(Self {
                    start: s0,
                    step,
                    count: 0,
                });
            }
            let span = s0 - e0;
            let abs_step = step.unsigned_abs();
            let count = (span / abs_step).saturating_add(1);
            Ok(Self {
                start: s0,
                step,
                count,
            })
        }
    }

    #[must_use]
    pub const fn resolve_index(&self, idx: usize) -> Option<usize> {
        if idx >= self.count {
            return None;
        }
        let offset = (idx as isize).saturating_mul(self.step);
        let target = (self.start as isize).saturating_add(offset);
        if target < 0 {
            None
        } else {
            Some(target as usize)
        }
    }
}

/// Zero-copy strided view over a `Tensor<T>`.
/// Unlike MATLAB (which copies memory on `A(1:2:end, :)`), `TensorView` allocates 0 bytes!
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a, T: Clone> {
    source: &'a Tensor<T>,
    row_slice: SliceRange,
    col_slice: SliceRange,
}

impl<'a, T: Clone + Copy + PartialEq> TensorView<'a, T> {
    pub fn new(
        source: &'a Tensor<T>,
        row_slice: SliceRange,
        col_slice: SliceRange,
    ) -> Result<Self, TensorError> {
        if row_slice.count > 0 {
            let last_r = row_slice
                .resolve_index(row_slice.count - 1)
                .ok_or(TensorError::InvalidSliceRange {
                    start: row_slice.start,
                    step: row_slice.step,
                    stop: 0,
                    len: source.rows(),
                })?;
            if row_slice.start >= source.rows() || last_r >= source.rows() {
                return Err(TensorError::IndexOutOfBounds {
                    index: vec![last_r, 0],
                    shape: source.shape().to_vec(),
                });
            }
        }
        if col_slice.count > 0 {
            let last_c = col_slice
                .resolve_index(col_slice.count - 1)
                .ok_or(TensorError::InvalidSliceRange {
                    start: col_slice.start,
                    step: col_slice.step,
                    stop: 0,
                    len: source.cols(),
                })?;
            if col_slice.start >= source.cols() || last_c >= source.cols() {
                return Err(TensorError::IndexOutOfBounds {
                    index: vec![0, last_c],
                    shape: source.shape().to_vec(),
                });
            }
        }
        Ok(Self {
            source,
            row_slice,
            col_slice,
        })
    }

    #[must_use]
    pub const fn rows(&self) -> usize {
        self.row_slice.count
    }

    #[must_use]
    pub const fn cols(&self) -> usize {
        self.col_slice.count
    }

    #[must_use]
    pub const fn shape(&self) -> [usize; 2] {
        [self.row_slice.count, self.col_slice.count]
    }

    pub fn get(&self, r: usize, c: usize) -> Result<T, TensorError> {
        let src_r =
            self.row_slice
                .resolve_index(r)
                .ok_or_else(|| TensorError::IndexOutOfBounds {
                    index: vec![r, c],
                    shape: vec![self.rows(), self.cols()],
                })?;
        let src_c =
            self.col_slice
                .resolve_index(c)
                .ok_or_else(|| TensorError::IndexOutOfBounds {
                    index: vec![r, c],
                    shape: vec![self.rows(), self.cols()],
                })?;
        self.source.get(src_r, src_c)
    }

    /// Materializes this view into an owned `Tensor<T>` when needed.
    pub fn to_owned_tensor(&self) -> Result<Tensor<T>, TensorError> {
        let r_cnt = self.rows();
        let c_cnt = self.cols();
        let mut data = Vec::with_capacity(r_cnt.saturating_mul(c_cnt));
        for c in 0..c_cnt {
            for r in 0..r_cnt {
                data.push(self.get(r, c)?);
            }
        }
        Tensor::from_column_major(
            r_cnt,
            c_cnt,
            data,
            self.source.unit(),
            self.source.budget(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_core::{MemoryBudget, SiUnit};

    #[test]
    fn test_zero_copy_strided_view_and_logical_mask() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        // 4x4 matrix with values 1..=16
        let vals: Vec<f64> = (1..=16).map(f64::from).collect();
        let a = Tensor::from_column_major(4, 4, vals, SiUnit::METER, &budget)?;

        // Zero-copy view A(1:2:4, 2:4) in 1-based indexing
        let r_slice = SliceRange::matlab_1based(1, 2, 4, 4)?;
        let c_slice = SliceRange::matlab_1based(2, 1, 4, 4)?;
        let view = a.view(r_slice, c_slice)?;
        assert_eq!(view.shape(), [2, 3]);
        assert_eq!(view.get(0, 0)?, a.get_1based(1, 2)?);
        assert_eq!(view.get(1, 2)?, a.get_1based(3, 4)?);

        // Logical mask indexing: extract elements > 12_m
        let threshold = Tensor::scalar(12.0, SiUnit::METER, &budget)?;
        let mask = a.compare_mask(&threshold, ">", |x, y| x > y)?;
        let filtered = a.filter_by_mask(&mask)?;
        assert_eq!(filtered.rows(), 4);
        assert_eq!(filtered.sum_all()?, 13.0 + 14.0 + 15.0 + 16.0);
        Ok(())
    }
}
