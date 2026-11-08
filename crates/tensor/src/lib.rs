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

pub mod broadcast;
pub mod containers;
pub mod nd;
pub mod sparse;
pub mod view;

pub use broadcast::broadcast_elementwise;
pub use containers::{CellArray, CellValue, StructArray};
pub use nd::NdTensor;

pub use sparse::SparseCsc;
pub use view::{SliceRange, TensorView};

use num_core::{AllocError, AllocReservation, MemoryBudget, SiUnit, UnitError};
use std::sync::Arc;
use thiserror::Error;

pub const TILE_DIM: usize = 64;
pub const TILE_LEN: usize = TILE_DIM * TILE_DIM; // 4096 elements per 2D tile (32 KB for f64)
pub const FLAT_THRESHOLD_ELEMENTS: usize = 8192;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum TensorError {
    #[error("Allocation error: {0}")]
    Alloc(#[from] AllocError),
    #[error("Unit error: {0}")]
    Unit(#[from] UnitError),
    #[error("Index {index:?} is out of bounds for tensor with shape {shape:?}")]
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
    #[error("Shape mismatch for `{op}`: left is {lhs:?}, right is {rhs:?}")]
    ShapeMismatch {
        op: &'static str,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    #[error("Cannot reshape tensor of shape {from:?} ({numel} elements) into {to:?}")]
    InvalidReshape {
        from: [usize; 2],
        to: [usize; 2],
        numel: usize,
    },
    #[error("Invalid slice range {start}:{step}:{stop} for dimension of length {len}")]
    InvalidSliceRange {
        start: usize,
        step: isize,
        stop: usize,
        len: usize,
    },
    #[error("Data length {data_len} does not match shape {shape:?} (expected {expected})")]
    DataLengthMismatch {
        data_len: usize,
        shape: Vec<usize>,
        expected: usize,
    },
}

#[derive(Clone, Debug)]
pub enum TileData<T: Clone> {
    Uniform(T),
    Dense {
        values: Arc<Vec<T>>,
        _reservation: Arc<AllocReservation>,
    },
}

impl<T: Clone + PartialEq> PartialEq for TileData<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Uniform(a), Self::Uniform(b)) => a == b,
            (Self::Dense { values: a, .. }, Self::Dense { values: b, .. }) => a == b,
            (Self::Uniform(u), Self::Dense { values, .. })
            | (Self::Dense { values, .. }, Self::Uniform(u)) => values.iter().all(|v| v == u),
        }
    }
}

#[derive(Clone, Debug)]
pub enum StorageTier<T: Clone> {
    Flat {
        values: Arc<Vec<T>>,
        _reservation: Arc<AllocReservation>,
    },
    Tiled {
        tile_rows: usize,
        tile_cols: usize,
        tiles: Arc<Vec<TileData<T>>>,
    },
}

/// Two-tier Copy-on-Write Matrix/Tensor with physical `SiUnit` tracking.
#[derive(Clone, Debug)]
pub struct Tensor<T: Clone = f64> {
    shape: [usize; 2],
    unit: SiUnit,
    budget: Arc<MemoryBudget>,
    storage: StorageTier<T>,
}

impl<T: Clone + Copy + PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape || self.unit != other.unit {
            return false;
        }
        for c in 0..self.cols() {
            for r in 0..self.rows() {
                if self.get(r, c).ok() != other.get(r, c).ok() {
                    return false;
                }
            }
        }
        true
    }
}

impl<T: Clone + Copy + PartialEq> Tensor<T> {
    pub fn full_with_budget(
        rows: usize,
        cols: usize,
        fill_value: T,
        unit: SiUnit,
        budget: &Arc<MemoryBudget>,
    ) -> Result<Self, TensorError> {
        let elem_count = MemoryBudget::checked_element_count(&[rows, cols])?;
        if elem_count <= FLAT_THRESHOLD_ELEMENTS {
            let res = budget.try_reserve_elements(elem_count, std::mem::size_of::<T>())?;
            let values = vec![fill_value; elem_count];
            Ok(Self {
                shape: [rows, cols],
                unit,
                budget: Arc::clone(budget),
                storage: StorageTier::Flat {
                    values: Arc::new(values),
                    _reservation: Arc::new(res),
                },
            })
        } else {
            let tile_rows = rows.div_ceil(TILE_DIM);
            let tile_cols = cols.div_ceil(TILE_DIM);
            let num_tiles = MemoryBudget::checked_element_count(&[tile_rows, tile_cols])?;
            let _meta_res =
                budget.try_reserve_elements(num_tiles, std::mem::size_of::<TileData<T>>())?;
            let tiles = vec![TileData::Uniform(fill_value); num_tiles];
            Ok(Self {
                shape: [rows, cols],
                unit,
                budget: Arc::clone(budget),
                storage: StorageTier::Tiled {
                    tile_rows,
                    tile_cols,
                    tiles: Arc::new(tiles),
                },
            })
        }
    }

    pub fn from_column_major(
        rows: usize,
        cols: usize,
        data: Vec<T>,
        unit: SiUnit,
        budget: &Arc<MemoryBudget>,
    ) -> Result<Self, TensorError> {
        let expected = MemoryBudget::checked_element_count(&[rows, cols])?;
        if data.len() != expected {
            return Err(TensorError::DataLengthMismatch {
                data_len: data.len(),
                shape: vec![rows, cols],
                expected,
            });
        }
        let res = budget.try_reserve_elements(expected, std::mem::size_of::<T>())?;
        Ok(Self {
            shape: [rows, cols],
            unit,
            budget: Arc::clone(budget),
            storage: StorageTier::Flat {
                values: Arc::new(data),
                _reservation: Arc::new(res),
            },
        })
    }

    #[must_use]
    pub const fn rows(&self) -> usize {
        self.shape[0]
    }

    #[must_use]
    pub const fn cols(&self) -> usize {
        self.shape[1]
    }

    #[must_use]
    pub const fn shape(&self) -> [usize; 2] {
        self.shape
    }

    #[must_use]
    pub const fn numel(&self) -> usize {
        self.shape[0].saturating_mul(self.shape[1])
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.shape[0] == 0 || self.shape[1] == 0
    }

    #[must_use]
    pub const fn is_scalar(&self) -> bool {
        self.shape[0] == 1 && self.shape[1] == 1
    }

    #[must_use]
    pub const fn is_vector(&self) -> bool {
        self.shape[0] == 1 || self.shape[1] == 1
    }

    #[must_use]
    pub const fn unit(&self) -> SiUnit {
        self.unit
    }

    #[must_use]
    pub const fn budget(&self) -> &Arc<MemoryBudget> {
        &self.budget
    }

    pub fn set_unit(&mut self, unit: SiUnit) {
        self.unit = unit;
    }

    #[must_use]
    pub fn materialized_tile_count(&self) -> usize {
        match &self.storage {
            StorageTier::Flat { .. } => 0,
            StorageTier::Tiled { tiles, .. } => tiles
                .iter()
                .filter(|t| matches!(t, TileData::Dense { .. }))
                .count(),
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Result<T, TensorError> {
        let [rows, cols] = self.shape;
        if row >= rows || col >= cols {
            return Err(TensorError::IndexOutOfBounds {
                index: vec![row, col],
                shape: vec![rows, cols],
            });
        }
        match &self.storage {
            StorageTier::Flat { values, .. } => {
                let idx = col.saturating_mul(rows).saturating_add(row);
                values
                    .get(idx)
                    .copied()
                    .ok_or_else(|| TensorError::IndexOutOfBounds {
                        index: vec![row, col],
                        shape: vec![rows, cols],
                    })
            }
            StorageTier::Tiled {
                tile_rows, tiles, ..
            } => {
                let tr = row / TILE_DIM;
                let tc = col / TILE_DIM;
                let tile_idx = tc.saturating_mul(*tile_rows).saturating_add(tr);
                let tile =
                    tiles
                        .get(tile_idx)
                        .ok_or_else(|| TensorError::IndexOutOfBounds {
                            index: vec![row, col],
                            shape: vec![rows, cols],
                        })?;
                match tile {
                    TileData::Uniform(val) => Ok(*val),
                    TileData::Dense { values, .. } => {
                        let lr = row % TILE_DIM;
                        let lc = col % TILE_DIM;
                        let local_idx = lc.saturating_mul(TILE_DIM).saturating_add(lr);
                        values.get(local_idx).copied().ok_or_else(|| {
                            TensorError::IndexOutOfBounds {
                                index: vec![row, col],
                                shape: vec![rows, cols],
                            }
                        })
                    }
                }
            }
        }
    }

    /// 0-based linear column-major indexing (`0 <= linear_idx < numel()`).
    pub fn get_linear(&self, linear_idx: usize) -> Result<T, TensorError> {
        let rows = self.rows();
        if rows == 0 || linear_idx >= self.numel() {
            return Err(TensorError::IndexOutOfBounds {
                index: vec![linear_idx],
                shape: self.shape.to_vec(),
            });
        }
        let r = linear_idx % rows;
        let c = linear_idx / rows;
        self.get(r, c)
    }

    pub fn set_linear(&mut self, linear_idx: usize, val: T) -> Result<(), TensorError> {
        let rows = self.rows();
        if rows == 0 || linear_idx >= self.numel() {
            return Err(TensorError::IndexOutOfBounds {
                index: vec![linear_idx],
                shape: self.shape.to_vec(),
            });
        }
        let r = linear_idx % rows;
        let c = linear_idx / rows;
        self.set(r, c, val)
    }

    pub fn get_1based(&self, row1: usize, col1: usize) -> Result<T, TensorError> {
        if row1 == 0 || col1 == 0 {
            return Err(TensorError::IndexOutOfBounds {
                index: vec![row1, col1],
                shape: vec![self.rows(), self.cols()],
            });
        }
        self.get(row1 - 1, col1 - 1)
    }

    pub fn set(&mut self, row: usize, col: usize, val: T) -> Result<(), TensorError> {
        let [rows, cols] = self.shape;
        if row >= rows || col >= cols {
            return Err(TensorError::IndexOutOfBounds {
                index: vec![row, col],
                shape: vec![rows, cols],
            });
        }
        match &mut self.storage {
            StorageTier::Flat {
                values,
                _reservation,
            } => {
                let idx = col.saturating_mul(rows).saturating_add(row);
                if Arc::strong_count(values) > 1 {
                    let new_res = self
                        .budget
                        .try_reserve_elements(values.len(), std::mem::size_of::<T>())?;
                    *_reservation = Arc::new(new_res);
                }
                let vec_mut = Arc::make_mut(values);
                let slot = vec_mut
                    .get_mut(idx)
                    .ok_or_else(|| TensorError::IndexOutOfBounds {
                        index: vec![row, col],
                        shape: vec![rows, cols],
                    })?;
                *slot = val;
                Ok(())
            }
            StorageTier::Tiled {
                tile_rows, tiles, ..
            } => {
                let tr = row / TILE_DIM;
                let tc = col / TILE_DIM;
                let tile_idx = tc.saturating_mul(*tile_rows).saturating_add(tr);
                let lr = row % TILE_DIM;
                let lc = col % TILE_DIM;
                let local_idx = lc.saturating_mul(TILE_DIM).saturating_add(lr);

                let tiles_mut = Arc::make_mut(tiles);
                let tile = tiles_mut.get_mut(tile_idx).ok_or_else(|| {
                    TensorError::IndexOutOfBounds {
                        index: vec![row, col],
                        shape: vec![rows, cols],
                    }
                })?;

                match tile {
                    TileData::Uniform(u) => {
                        if *u == val {
                            return Ok(());
                        }
                        let res = self
                            .budget
                            .try_reserve_elements(TILE_LEN, std::mem::size_of::<T>())?;
                        let mut dense = vec![*u; TILE_LEN];
                        if let Some(slot) = dense.get_mut(local_idx) {
                            *slot = val;
                        }
                        *tile = TileData::Dense {
                            values: Arc::new(dense),
                            _reservation: Arc::new(res),
                        };
                    }
                    TileData::Dense {
                        values,
                        _reservation,
                    } => {
                        if Arc::strong_count(values) > 1 {
                            let res = self
                                .budget
                                .try_reserve_elements(TILE_LEN, std::mem::size_of::<T>())?;
                            *_reservation = Arc::new(res);
                        }
                        let dense_mut = Arc::make_mut(values);
                        let slot = dense_mut.get_mut(local_idx).ok_or_else(|| {
                            TensorError::IndexOutOfBounds {
                                index: vec![row, col],
                                shape: vec![rows, cols],
                            }
                        })?;
                        *slot = val;
                    }
                }
                Ok(())
            }
        }
    }

    pub fn set_1based(&mut self, row1: usize, col1: usize, val: T) -> Result<(), TensorError> {
        if row1 == 0 || col1 == 0 {
            return Err(TensorError::IndexOutOfBounds {
                index: vec![row1, col1],
                shape: vec![self.rows(), self.cols()],
            });
        }
        self.set(row1 - 1, col1 - 1, val)
    }

    pub fn to_column_major_vec(&self) -> Result<Vec<T>, TensorError> {
        let [rows, cols] = self.shape;
        let count = MemoryBudget::checked_element_count(&[rows, cols])?;
        let _res = self
            .budget
            .try_reserve_elements(count, std::mem::size_of::<T>())?;
        let mut out = Vec::with_capacity(count);
        for c in 0..cols {
            for r in 0..rows {
                out.push(self.get(r, c)?);
            }
        }
        Ok(out)
    }

    pub fn transpose(&self) -> Result<Self, TensorError> {
        let [rows, cols] = self.shape;
        let mut out = Vec::with_capacity(self.numel());
        for r in 0..rows {
            for c in 0..cols {
                out.push(self.get(r, c)?);
            }
        }
        Self::from_column_major(cols, rows, out, self.unit, &self.budget)
    }

    /// Reshapes the tensor to `[new_rows, new_cols]` preserving column-major element order.
    /// Zero-copy when storage is `StorageTier::Flat`!
    pub fn reshape(&self, new_rows: usize, new_cols: usize) -> Result<Self, TensorError> {
        let target_numel = MemoryBudget::checked_element_count(&[new_rows, new_cols])?;
        if target_numel != self.numel() {
            return Err(TensorError::InvalidReshape {
                from: self.shape,
                to: [new_rows, new_cols],
                numel: self.numel(),
            });
        }
        if let StorageTier::Flat {
            values,
            _reservation,
        } = &self.storage
        {
            return Ok(Self {
                shape: [new_rows, new_cols],
                unit: self.unit,
                budget: Arc::clone(&self.budget),
                storage: StorageTier::Flat {
                    values: Arc::clone(values),
                    _reservation: Arc::clone(_reservation),
                },
            });
        }
        let col_major = self.to_column_major_vec()?;
        Self::from_column_major(new_rows, new_cols, col_major, self.unit, &self.budget)
    }

    /// Creates a zero-copy strided `TensorView` (`A(r_start:r_step:r_stop, c_start:c_step:c_stop)`).
    pub fn view(
        &self,
        row_range: SliceRange,
        col_range: SliceRange,
    ) -> Result<TensorView<'_, T>, TensorError> {
        TensorView::new(self, row_range, col_range)
    }

    /// Extracts a sub-matrix by 0-based row and column index lists (`A(rows, cols)`).
    pub fn slice_by_indices(
        &self,
        row_indices: &[usize],
        col_indices: &[usize],
    ) -> Result<Self, TensorError> {
        let out_rows = row_indices.len();
        let out_cols = col_indices.len();
        let count = MemoryBudget::checked_element_count(&[out_rows, out_cols])?;
        let mut out = Vec::with_capacity(count);
        for &c in col_indices {
            for &r in row_indices {
                out.push(self.get(r, c)?);
            }
        }
        Self::from_column_major(out_rows, out_cols, out, self.unit, &self.budget)
    }

    /// Horizontal concatenation `[A, B]`.
    pub fn horzcat(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.rows() != rhs.rows() {
            return Err(TensorError::ShapeMismatch {
                op: "horzcat",
                lhs: self.shape.to_vec(),
                rhs: rhs.shape.to_vec(),
            });
        }
        let unit = self.unit.verify_same(rhs.unit, "horzcat")?;
        let mut col_major = self.to_column_major_vec()?;
        let rhs_vec = rhs.to_column_major_vec()?;
        col_major.extend(rhs_vec);
        Self::from_column_major(
            self.rows(),
            self.cols().saturating_add(rhs.cols()),
            col_major,
            unit,
            &self.budget,
        )
    }

    /// Vertical concatenation `[A; B]`.
    pub fn vertcat(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.cols() != rhs.cols() {
            return Err(TensorError::ShapeMismatch {
                op: "vertcat",
                lhs: self.shape.to_vec(),
                rhs: rhs.shape.to_vec(),
            });
        }
        let unit = self.unit.verify_same(rhs.unit, "vertcat")?;
        let new_rows = self.rows().saturating_add(rhs.rows());
        let cols = self.cols();
        let mut out = Vec::with_capacity(new_rows.saturating_mul(cols));
        for c in 0..cols {
            for r in 0..self.rows() {
                out.push(self.get(r, c)?);
            }
            for r in 0..rhs.rows() {
                out.push(rhs.get(r, c)?);
            }
        }
        Self::from_column_major(new_rows, cols, out, unit, &self.budget)
    }
}

impl Tensor<f64> {
    pub fn zeros(rows: usize, cols: usize, budget: &Arc<MemoryBudget>) -> Result<Self, TensorError> {
        Self::full_with_budget(rows, cols, 0.0, SiUnit::DIMENSIONLESS, budget)
    }

    pub fn ones(rows: usize, cols: usize, budget: &Arc<MemoryBudget>) -> Result<Self, TensorError> {
        Self::full_with_budget(rows, cols, 1.0, SiUnit::DIMENSIONLESS, budget)
    }

    pub fn scalar(value: f64, unit: SiUnit, budget: &Arc<MemoryBudget>) -> Result<Self, TensorError> {
        Self::from_column_major(1, 1, vec![value], unit, budget)
    }

    pub fn eye(n: usize, budget: &Arc<MemoryBudget>) -> Result<Self, TensorError> {
        let mut m = Self::zeros(n, n, budget)?;
        for i in 0..n {
            m.set(i, i, 1.0)?;
        }
        Ok(m)
    }

    pub fn linspace(
        start: f64,
        stop: f64,
        n: usize,
        unit: SiUnit,
        budget: &Arc<MemoryBudget>,
    ) -> Result<Self, TensorError> {
        if n == 0 {
            return Self::from_column_major(1, 0, Vec::new(), unit, budget);
        }
        if n == 1 {
            return Self::from_column_major(1, 1, vec![stop], unit, budget);
        }
        let step = (stop - start) / ((n - 1) as f64);
        let data: Vec<f64> = (0..n).map(|i| start + (i as f64) * step).collect();
        Self::from_column_major(1, n, data, unit, budget)
    }

    pub fn checked_add(&self, rhs: &Self) -> Result<Self, TensorError> {
        let unit = self.unit.verify_same(rhs.unit, "+")?;
        self.binary_op(rhs, unit, "+", |a, b| a + b)
    }

    pub fn checked_sub(&self, rhs: &Self) -> Result<Self, TensorError> {
        let unit = self.unit.verify_same(rhs.unit, "-")?;
        self.binary_op(rhs, unit, "-", |a, b| a - b)
    }

    pub fn elem_mul(&self, rhs: &Self) -> Result<Self, TensorError> {
        let unit = self.unit.checked_mul(rhs.unit)?;
        self.binary_op(rhs, unit, ".*", |a, b| a * b)
    }

    pub fn elem_div(&self, rhs: &Self) -> Result<Self, TensorError> {
        let unit = self.unit.checked_div(rhs.unit)?;
        self.binary_op(rhs, unit, "./", |a, b| a / b)
    }

    /// Elementwise comparison returning a 0.0 / 1.0 dimensionless mask tensor.
    pub fn compare_mask<F>(&self, rhs: &Self, op_name: &'static str, cmp: F) -> Result<Self, TensorError>
    where
        F: Fn(f64, f64) -> bool,
    {
        self.unit.verify_same(rhs.unit, op_name)?;
        self.binary_op(rhs, SiUnit::DIMENSIONLESS, op_name, |a, b| {
            if cmp(a, b) {
                1.0
            } else {
                0.0
            }
        })
    }

    /// Logical mask indexing (`A(mask)`): extracts all elements where `mask != 0.0` as a column vector.
    pub fn filter_by_mask(&self, mask: &Self) -> Result<Self, TensorError> {
        if self.shape != mask.shape {
            return Err(TensorError::ShapeMismatch {
                op: "logical indexing",
                lhs: self.shape.to_vec(),
                rhs: mask.shape.to_vec(),
            });
        }
        let mut selected = Vec::new();
        for c in 0..self.cols() {
            for r in 0..self.rows() {
                if mask.get(r, c)? != 0.0 {
                    selected.push(self.get(r, c)?);
                }
            }
        }
        let len = selected.len();
        Self::from_column_major(len, 1, selected, self.unit, &self.budget)
    }

    /// Reductions: sum, mean, min, max, std, Frobenius norm.
    pub fn sum_all(&self) -> Result<f64, TensorError> {
        let mut total = 0.0;
        for c in 0..self.cols() {
            for r in 0..self.rows() {
                total += self.get(r, c)?;
            }
        }
        Ok(total)
    }

    pub fn mean_all(&self) -> Result<f64, TensorError> {
        if self.is_empty() {
            return Ok(f64::NAN);
        }
        Ok(self.sum_all()? / (self.numel() as f64))
    }

    pub fn min_all(&self) -> Result<f64, TensorError> {
        let mut best = f64::INFINITY;
        for c in 0..self.cols() {
            for r in 0..self.rows() {
                best = best.min(self.get(r, c)?);
            }
        }
        Ok(best)
    }

    pub fn max_all(&self) -> Result<f64, TensorError> {
        let mut best = f64::NEG_INFINITY;
        for c in 0..self.cols() {
            for r in 0..self.rows() {
                best = best.max(self.get(r, c)?);
            }
        }
        Ok(best)
    }

    pub fn std_all(&self) -> Result<f64, TensorError> {
        let n = self.numel();
        if n <= 1 {
            return Ok(0.0);
        }
        let mu = self.mean_all()?;
        let mut sq = 0.0;
        for c in 0..self.cols() {
            for r in 0..self.rows() {
                let d = self.get(r, c)? - mu;
                sq += d * d;
            }
        }
        Ok((sq / ((n - 1) as f64)).sqrt())
    }

    pub fn norm_frobenius(&self) -> Result<f64, TensorError> {
        let mut acc: f64 = 0.0;
        for c in 0..self.cols() {
            for r in 0..self.rows() {
                acc = acc.hypot(self.get(r, c)?);
            }
        }
        Ok(acc)
    }

    pub fn map_unary<F>(&self, new_unit: SiUnit, f: F) -> Result<Self, TensorError>
    where
        F: Fn(f64) -> f64,
    {
        match &self.storage {
            StorageTier::Flat { values, .. } => {
                let mapped: Vec<f64> = values.iter().map(|&x| f(x)).collect();
                Self::from_column_major(self.rows(), self.cols(), mapped, new_unit, &self.budget)
            }
            StorageTier::Tiled {
                tile_rows,
                tile_cols,
                tiles,
            } => {
                let mut new_tiles = Vec::with_capacity(tiles.len());
                for tile in tiles.iter() {
                    match tile {
                        TileData::Uniform(u) => new_tiles.push(TileData::Uniform(f(*u))),
                        TileData::Dense { values, .. } => {
                            let res = self
                                .budget
                                .try_reserve_elements(TILE_LEN, std::mem::size_of::<f64>())?;
                            let mapped: Vec<f64> = values.iter().map(|&x| f(x)).collect();
                            new_tiles.push(TileData::Dense {
                                values: Arc::new(mapped),
                                _reservation: Arc::new(res),
                            });
                        }
                    }
                }
                Ok(Self {
                    shape: self.shape,
                    unit: new_unit,
                    budget: Arc::clone(&self.budget),
                    storage: StorageTier::Tiled {
                        tile_rows: *tile_rows,
                        tile_cols: *tile_cols,
                        tiles: Arc::new(new_tiles),
                    },
                })
            }
        }
    }

    fn binary_op<F>(
        &self,
        rhs: &Self,
        out_unit: SiUnit,
        op_name: &'static str,
        op: F,
    ) -> Result<Self, TensorError>
    where
        F: Fn(f64, f64) -> f64,
    {
        if self.is_scalar() {
            let s = self.get(0, 0)?;
            return rhs.map_unary(out_unit, |b| op(s, b));
        }
        if rhs.is_scalar() {
            let s = rhs.get(0, 0)?;
            return self.map_unary(out_unit, |a| op(a, s));
        }
        if self.shape != rhs.shape {
            return Err(TensorError::ShapeMismatch {
                op: op_name,
                lhs: self.shape.to_vec(),
                rhs: rhs.shape.to_vec(),
            });
        }
        if let (
            StorageTier::Tiled {
                tile_rows,
                tile_cols,
                tiles: l_tiles,
            },
            StorageTier::Tiled { tiles: r_tiles, .. },
        ) = (&self.storage, &rhs.storage)
        {
            let mut out_tiles = Vec::with_capacity(l_tiles.len());
            for (lt, rt) in l_tiles.iter().zip(r_tiles.iter()) {
                match (lt, rt) {
                    (TileData::Uniform(a), TileData::Uniform(b)) => {
                        out_tiles.push(TileData::Uniform(op(*a, *b)));
                    }
                    (TileData::Uniform(a), TileData::Dense { values: bv, .. }) => {
                        let res = self
                            .budget
                            .try_reserve_elements(TILE_LEN, std::mem::size_of::<f64>())?;
                        let v: Vec<f64> = bv.iter().map(|&b| op(*a, b)).collect();
                        out_tiles.push(TileData::Dense {
                            values: Arc::new(v),
                            _reservation: Arc::new(res),
                        });
                    }
                    (TileData::Dense { values: av, .. }, TileData::Uniform(b)) => {
                        let res = self
                            .budget
                            .try_reserve_elements(TILE_LEN, std::mem::size_of::<f64>())?;
                        let v: Vec<f64> = av.iter().map(|&a| op(a, *b)).collect();
                        out_tiles.push(TileData::Dense {
                            values: Arc::new(v),
                            _reservation: Arc::new(res),
                        });
                    }
                    (TileData::Dense { values: av, .. }, TileData::Dense { values: bv, .. }) => {
                        let res = self
                            .budget
                            .try_reserve_elements(TILE_LEN, std::mem::size_of::<f64>())?;
                        let v: Vec<f64> =
                            av.iter().zip(bv.iter()).map(|(&a, &b)| op(a, b)).collect();
                        out_tiles.push(TileData::Dense {
                            values: Arc::new(v),
                            _reservation: Arc::new(res),
                        });
                    }
                }
            }
            return Ok(Self {
                shape: self.shape,
                unit: out_unit,
                budget: Arc::clone(&self.budget),
                storage: StorageTier::Tiled {
                    tile_rows: *tile_rows,
                    tile_cols: *tile_cols,
                    tiles: Arc::new(out_tiles),
                },
            });
        }

        let [rows, cols] = self.shape;
        let mut out = Vec::with_capacity(self.numel());
        for c in 0..cols {
            for r in 0..rows {
                out.push(op(self.get(r, c)?, rhs.get(r, c)?));
            }
        }
        Self::from_column_major(rows, cols, out, out_unit, &self.budget)
    }
}
