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

//! Heterogeneous cell arrays and multi-element dynamic struct arrays.
//! Provides MATLAB-style `{}` container semantics and `s.field` record structures.

use crate::{Tensor, TensorError};
use num_core::{MemoryBudget, SiUnit};
use std::collections::BTreeMap;
use std::sync::Arc;

/// A heterogeneous value that can reside inside a MATLAB Cell or Struct field.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum CellValue {
    #[default]
    Empty,
    Float(f64),
    String(String),
    Matrix(Tensor<f64>),
    Cell(Box<CellArray>),
    Struct(Box<StructArray>),
}

/// A 2D heterogeneous Cell Array with column-major MATLAB indexing.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CellArray {
    rows: usize,
    cols: usize,
    data: Vec<CellValue>,
}

impl CellArray {
    /// Creates a new `rows` × `cols` cell array initialized with `CellValue::Empty`.
    #[must_use]
    pub fn new(rows: usize, cols: usize) -> Self {
        let total = rows.saturating_mul(cols);
        let mut data = Vec::with_capacity(total);
        for _ in 0..total {
            data.push(CellValue::Empty);
        }
        Self { rows, cols, data }
    }

    /// Returns the `[rows, cols]` shape.
    #[must_use]
    pub const fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    /// Returns the number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Total number of cells.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }

    /// Linear index in column-major order (Fortran convention).
    fn linear_index(&self, r: usize, c: usize) -> Result<usize, TensorError> {
        if r >= self.rows || c >= self.cols {
            return Err(TensorError::IndexOutOfBounds {
                index: vec![r, c],
                shape: vec![self.rows, self.cols],
            });
        }
        Ok(c.saturating_mul(self.rows).saturating_add(r))
    }

    /// Get a reference to the cell value at `(r, c)`.
    pub fn get(&self, r: usize, c: usize) -> Result<&CellValue, TensorError> {
        let idx = self.linear_index(r, c)?;
        self.data.get(idx).ok_or(TensorError::IndexOutOfBounds {
            index: vec![r, c],
            shape: vec![self.rows, self.cols],
        })
    }

    /// Set the cell value at `(r, c)`.
    pub fn set(&mut self, r: usize, c: usize, val: CellValue) -> Result<(), TensorError> {
        let idx = self.linear_index(r, c)?;
        if let Some(slot) = self.data.get_mut(idx) {
            *slot = val;
            Ok(())
        } else {
            Err(TensorError::IndexOutOfBounds {
                index: vec![r, c],
                shape: vec![self.rows, self.cols],
            })
        }
    }

    /// Convert a cell array of scalar floats or 1x1 tensors into a contiguous numerical `Tensor<f64>`.
    /// Emulates MATLAB `cell2mat`.
    pub fn cell2mat(&self, budget: &Arc<MemoryBudget>) -> Result<Tensor<f64>, TensorError> {
        let mut out = Vec::with_capacity(self.numel());
        for c in 0..self.cols {
            for r in 0..self.rows {
                let cell = self.get(r, c)?;
                let val = match cell {
                    CellValue::Float(f) => *f,
                    CellValue::Matrix(t) if t.rows() == 1 && t.cols() == 1 => t.get(0, 0)?,
                    CellValue::Empty => 0.0,
                    _ => {
                        return Err(TensorError::ShapeMismatch {
                            op: "cell2mat",
                            lhs: vec![self.rows, self.cols],
                            rhs: vec![1, 1],
                        });
                    }
                };
                out.push(val);
            }
        }
        Tensor::from_column_major(
            self.rows,
            self.cols,
            out,
            SiUnit::DIMENSIONLESS,
            budget,
        )
    }
}

/// A multi-element Struct Array where each field stores a vector of values of length `num_elements`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct StructArray {
    num_elements: usize,
    fields: BTreeMap<String, Vec<CellValue>>,
}

impl StructArray {
    /// Creates a struct array with `num_elements` elements and the given field names.
    #[must_use]
    pub fn new(num_elements: usize, field_names: &[&str]) -> Self {
        let mut fields = BTreeMap::new();
        for &name in field_names {
            let mut col = Vec::with_capacity(num_elements);
            for _ in 0..num_elements {
                col.push(CellValue::Empty);
            }
            fields.insert(name.to_string(), col);
        }
        Self {
            num_elements,
            fields,
        }
    }

    /// Number of struct records in this array.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.num_elements
    }

    /// Returns `true` if this struct array has 0 elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.num_elements == 0
    }

    /// Returns the list of field names.
    #[must_use]
    pub fn fieldnames(&self) -> Vec<String> {
        self.fields.keys().cloned().collect()
    }

    /// Returns true if `field` is present.
    #[must_use]
    pub fn isfield(&self, field: &str) -> bool {
        self.fields.contains_key(field)
    }

    /// Gets field value for struct record at `elem_idx`.
    pub fn get_field(&self, elem_idx: usize, field: &str) -> Option<&CellValue> {
        if elem_idx >= self.num_elements {
            return None;
        }
        self.fields.get(field)?.get(elem_idx)
    }

    /// Sets field value for struct record at `elem_idx`. If field does not exist, it is created.
    pub fn set_field(
        &mut self,
        elem_idx: usize,
        field: &str,
        val: CellValue,
    ) -> Result<(), TensorError> {
        if elem_idx >= self.num_elements {
            // Grow struct array up to elem_idx + 1
            let new_len = elem_idx.saturating_add(1);
            for vec in self.fields.values_mut() {
                while vec.len() < new_len {
                    vec.push(CellValue::Empty);
                }
            }
            self.num_elements = new_len;
        }

        let n = self.num_elements;
        let col = self.fields.entry(field.to_string()).or_insert_with(|| {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(CellValue::Empty);
            }
            v
        });

        if let Some(slot) = col.get_mut(elem_idx) {
            *slot = val;
            Ok(())
        } else {
            Err(TensorError::IndexOutOfBounds {
                index: vec![elem_idx],
                shape: vec![self.num_elements],
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_array_get_set_and_cell2mat() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        let mut cells = CellArray::new(2, 3);
        assert_eq!(cells.shape(), [2, 3]);
        assert_eq!(cells.numel(), 6);

        cells.set(0, 0, CellValue::Float(10.0))?;
        cells.set(1, 0, CellValue::Float(20.0))?;
        cells.set(0, 1, CellValue::Float(30.0))?;
        cells.set(1, 1, CellValue::Float(40.0))?;
        cells.set(0, 2, CellValue::Float(50.0))?;
        cells.set(1, 2, CellValue::Float(60.0))?;

        let mat = cells.cell2mat(&budget)?;
        assert_eq!(mat.shape(), [2, 3]);
        assert_eq!(mat.get(0, 0)?, 10.0);
        assert_eq!(mat.get(1, 1)?, 40.0);
        assert_eq!(mat.get(1, 2)?, 60.0);
        Ok(())
    }

    #[test]
    fn test_heterogeneous_cell_array() -> Result<(), Box<dyn std::error::Error>> {
        let mut cells = CellArray::new(2, 2);
        cells.set(0, 0, CellValue::String("hello".to_string()))?;
        cells.set(1, 0, CellValue::Float(3.14159))?;
        let inner = Box::new(CellArray::new(1, 1));
        cells.set(0, 1, CellValue::Cell(inner))?;

        match cells.get(0, 0)? {
            CellValue::String(s) => assert_eq!(s, "hello"),
            _ => return Err("Expected String".into()),
        }
        match cells.get(1, 0)? {
            CellValue::Float(f) => assert!((f - 3.14159).abs() < 1e-5),
            _ => return Err("Expected Float".into()),
        }
        Ok(())
    }

    #[test]
    fn test_struct_array_fields_and_growth() -> Result<(), Box<dyn std::error::Error>> {
        let mut s = StructArray::new(2, &["name", "score"]);
        assert_eq!(s.len(), 2);
        assert!(s.isfield("name"));
        assert!(s.isfield("score"));
        assert!(!s.isfield("missing"));

        s.set_field(0, "name", CellValue::String("Alice".to_string()))?;
        s.set_field(0, "score", CellValue::Float(98.5))?;
        s.set_field(1, "name", CellValue::String("Bob".to_string()))?;
        s.set_field(1, "score", CellValue::Float(85.0))?;

        // Dynamic growth to index 3 (4 elements total)
        s.set_field(3, "age", CellValue::Float(25.0))?;
        assert_eq!(s.len(), 4);
        assert!(s.isfield("age"));

        match s.get_field(0, "name") {
            Some(CellValue::String(name)) => assert_eq!(name, "Alice"),
            _ => return Err("Expected Alice".into()),
        }
        match s.get_field(3, "age") {
            Some(CellValue::Float(age)) => assert_eq!(*age, 25.0),
            _ => return Err("Expected 25.0".into()),
        }
        // Unset fields in grown record are empty
        assert_eq!(s.get_field(2, "name"), Some(&CellValue::Empty));
        Ok(())
    }
}
