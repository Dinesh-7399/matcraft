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

pub mod hdf5_stub;
pub mod mat73_chunked;

pub use hdf5_stub::{read_hdf5_superblock, HdfSuperblock};
pub use mat73_chunked::{query_element_out_of_core, read_chunked_archive, write_chunked_archive};

use num_core::{AllocError, MemoryBudget, SiUnit};
use std::collections::BTreeMap;
use std::sync::Arc;
use tensor::{Tensor, TensorError};
use thiserror::Error;

const MAT5_HEADER_LEN: usize = 128;
const MI_INT8: u32 = 1;
const MI_UINT32: u32 = 6;
const MI_INT32: u32 = 5;
const MI_DOUBLE: u32 = 9;
const MI_MATRIX: u32 = 14;
const MX_DOUBLE_CLASS: u32 = 6;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum FormatError {
    #[error("Allocation budget error: {0}")]
    Alloc(#[from] AllocError),
    #[error("Tensor error: {0}")]
    Tensor(#[from] TensorError),
    #[error("Truncated or malformed .mat file: {0}")]
    MalformedMat(String),
    #[error("Unsupported .mat endian indicator `{0:?}`")]
    UnsupportedEndian([u8; 2]),
    #[error("Malformed CSV at line {line}: {reason}")]
    MalformedCsv { line: usize, reason: String },
}

/// Clean-room MATLAB Level 5 (`.mat`) binary serializer.
/// Writes a standard 128-byte header followed by `miMATRIX` elements for each variable.
pub fn write_mat5(vars: &BTreeMap<String, Tensor<f64>>) -> Result<Vec<u8>, FormatError> {
    let mut out = Vec::with_capacity(MAT5_HEADER_LEN + vars.len() * 128);

    // 116-byte descriptive text + 8-byte subsys offset + 2-byte version (0x0100) + 2-byte endian ("IM")
    let desc = b"MATLAB 5.0 MAT-file, Platform: MatCraft Pure-Rust Engine, Created offline";
    out.extend_from_slice(desc);
    out.resize(124, b' ');
    out.extend_from_slice(&0x0100u16.to_le_bytes());
    out.extend_from_slice(b"IM");

    for (name, tensor) in vars {
        let mut matrix_payload = Vec::new();

        // 1. Array Flags Sub-element (miUINT32, 8 bytes)
        matrix_payload.extend_from_slice(&MI_UINT32.to_le_bytes());
        matrix_payload.extend_from_slice(&8u32.to_le_bytes());
        matrix_payload.extend_from_slice(&MX_DOUBLE_CLASS.to_le_bytes());
        matrix_payload.extend_from_slice(&0u32.to_le_bytes());

        // 2. Dimensions Array Sub-element (miINT32, 8 bytes: [rows, cols])
        matrix_payload.extend_from_slice(&MI_INT32.to_le_bytes());
        matrix_payload.extend_from_slice(&8u32.to_le_bytes());
        let rows_i32 = i32::try_from(tensor.rows())
            .map_err(|_| FormatError::MalformedMat("Row dimension exceeds i32::MAX".into()))?;
        let cols_i32 = i32::try_from(tensor.cols())
            .map_err(|_| FormatError::MalformedMat("Col dimension exceeds i32::MAX".into()))?;
        matrix_payload.extend_from_slice(&rows_i32.to_le_bytes());
        matrix_payload.extend_from_slice(&cols_i32.to_le_bytes());

        // 3. Array Name Sub-element (miINT8, padded to 8-byte boundary)
        let name_bytes = name.as_bytes();
        let name_len = u32::try_from(name_bytes.len())
            .map_err(|_| FormatError::MalformedMat("Variable name too long".into()))?;
        matrix_payload.extend_from_slice(&MI_INT8.to_le_bytes());
        matrix_payload.extend_from_slice(&name_len.to_le_bytes());
        matrix_payload.extend_from_slice(name_bytes);
        while matrix_payload.len() % 8 != 0 {
            matrix_payload.push(0);
        }

        // 4. Real Data Sub-element (miDOUBLE, 8 * numel bytes)
        let col_major = tensor.to_column_major_vec()?;
        let data_byte_len = u32::try_from(col_major.len().saturating_mul(8))
            .map_err(|_| FormatError::MalformedMat("Matrix payload exceeds 4GB".into()))?;
        matrix_payload.extend_from_slice(&MI_DOUBLE.to_le_bytes());
        matrix_payload.extend_from_slice(&data_byte_len.to_le_bytes());
        for val in col_major {
            matrix_payload.extend_from_slice(&val.to_le_bytes());
        }
        while matrix_payload.len() % 8 != 0 {
            matrix_payload.push(0);
        }

        // Wrap in outer miMATRIX tag
        let total_len = u32::try_from(matrix_payload.len())
            .map_err(|_| FormatError::MalformedMat("miMATRIX element exceeds 4GB".into()))?;
        out.extend_from_slice(&MI_MATRIX.to_le_bytes());
        out.extend_from_slice(&total_len.to_le_bytes());
        out.extend_from_slice(&matrix_payload);
    }

    Ok(out)
}

/// Clean-room MATLAB Level 5 (`.mat`) binary reader with strict hostile-header allocation guards.
pub fn read_mat5(
    bytes: &[u8],
    budget: &Arc<MemoryBudget>,
) -> Result<BTreeMap<String, Tensor<f64>>, FormatError> {
    if bytes.len() < MAT5_HEADER_LEN {
        return Err(FormatError::MalformedMat(
            "File shorter than 128-byte MAT v5 header".into(),
        ));
    }
    let endian = [
        bytes.get(126).copied().unwrap_or(0),
        bytes.get(127).copied().unwrap_or(0),
    ];
    if &endian != b"IM" {
        return Err(FormatError::UnsupportedEndian(endian));
    }

    let mut cursor = MAT5_HEADER_LEN;
    let mut vars = BTreeMap::new();

    while cursor < bytes.len() {
        if cursor.saturating_add(8) > bytes.len() {
            return Err(FormatError::MalformedMat(
                "Truncated data element tag".into(),
            ));
        }
        let tag_type = read_u32_le(bytes, cursor)?;
        let tag_len = read_u32_le(bytes, cursor.saturating_add(4))? as usize;
        cursor = cursor.saturating_add(8);

        if tag_type != MI_MATRIX {
            return Err(FormatError::MalformedMat(format!(
                "Expected miMATRIX (14), found tag {tag_type}"
            )));
        }
        let elem_end = cursor.checked_add(tag_len).ok_or_else(|| {
            FormatError::MalformedMat("Overflow computing miMATRIX end offset".into())
        })?;
        if elem_end > bytes.len() {
            return Err(FormatError::MalformedMat(
                "miMATRIX length exceeds file size (hostile header)".into(),
            ));
        }

        let (name, tensor) = parse_mi_matrix(bytes, cursor, elem_end, budget)?;
        vars.insert(name, tensor);
        cursor = align8(elem_end);
    }

    Ok(vars)
}

fn parse_mi_matrix(
    bytes: &[u8],
     mut pos: usize,
    end: usize,
    budget: &Arc<MemoryBudget>,
) -> Result<(String, Tensor<f64>), FormatError> {
    // 1. Array flags
    let (flags_type, flags_len, next_pos) = read_subtag(bytes, pos, end)?;
    if flags_type != MI_UINT32 || flags_len < 8 {
        return Err(FormatError::MalformedMat("Invalid array flags".into()));
    }
    pos = align8(next_pos.saturating_add(flags_len));

    // 2. Dimensions
    let (dim_type, dim_len, dim_pos) = read_subtag(bytes, pos, end)?;
    if dim_type != MI_INT32 || dim_len < 8 {
        return Err(FormatError::MalformedMat("Invalid dimensions array".into()));
    }
    let rows_i32 = read_i32_le(bytes, dim_pos)?;
    let cols_i32 = read_i32_le(bytes, dim_pos.saturating_add(4))?;
    if rows_i32 < 0 || cols_i32 < 0 {
        return Err(FormatError::MalformedMat(
            "Negative matrix dimensions in .mat header".into(),
        ));
    }
    let rows = rows_i32 as usize;
    let cols = cols_i32 as usize;
    let elem_count = MemoryBudget::checked_element_count(&[rows, cols])?;
    pos = align8(dim_pos.saturating_add(dim_len));

    // 3. Name
    let (name_type, name_len, name_pos) = read_subtag(bytes, pos, end)?;
    if name_type != MI_INT8 {
        return Err(FormatError::MalformedMat("Invalid variable name tag".into()));
    }
    let name_end = name_pos.saturating_add(name_len);
    let name_slice = bytes
        .get(name_pos..name_end)
        .ok_or_else(|| FormatError::MalformedMat("Truncated variable name".into()))?;
    let name = String::from_utf8(name_slice.to_vec())
        .map_err(|_| FormatError::MalformedMat("Non-UTF8 variable name".into()))?;
    pos = align8(name_end);

    // 4. Real Data
    let (data_type, data_len, data_pos) = read_subtag(bytes, pos, end)?;
    if data_type != MI_DOUBLE {
        return Err(FormatError::MalformedMat(format!(
            "Unsupported numeric class tag {data_type}"
        )));
    }
    let expected_bytes = elem_count.checked_mul(8).ok_or_else(|| {
        FormatError::MalformedMat("Overflow computing matrix byte length".into())
    })?;
    if data_len != expected_bytes {
        return Err(FormatError::MalformedMat(
            "Data payload length does not match dimensions".into(),
        ));
    }
    let data_end = data_pos.saturating_add(data_len);
    if data_end > end {
        return Err(FormatError::MalformedMat(
            "Truncated double payload in miMATRIX".into(),
        ));
    }

    // Reserve budget BEFORE allocating Vec<f64>
    let _res = budget.try_reserve_elements(elem_count, 8)?;
    let mut values = Vec::with_capacity(elem_count);
    for i in 0..elem_count {
        let offset = data_pos.saturating_add(i.saturating_mul(8));
        values.push(read_f64_le(bytes, offset)?);
    }

    let tensor = Tensor::from_column_major(rows, cols, values, SiUnit::DIMENSIONLESS, budget)?;
    Ok((name, tensor))
}

fn read_subtag(bytes: &[u8], pos: usize, end: usize) -> Result<(u32, usize, usize), FormatError> {
    if pos.saturating_add(8) > end {
        return Err(FormatError::MalformedMat("Truncated sub-element tag".into()));
    }
    let raw_type = read_u32_le(bytes, pos)?;
    // Handle Level 5 Small Data Element (SDE) format where upper 16 bits hold byte count (1..=4)
    if (raw_type >> 16) != 0 {
        let sde_bytes = (raw_type >> 16) as usize;
        let sde_type = raw_type & 0xFFFF;
        Ok((sde_type, sde_bytes, pos.saturating_add(4)))
    } else {
        let len = read_u32_le(bytes, pos.saturating_add(4))? as usize;
        Ok((raw_type, len, pos.saturating_add(8)))
    }
}

const fn align8(offset: usize) -> usize {
    let rem = offset % 8;
    if rem == 0 {
        offset
    } else {
        offset.saturating_add(8 - rem)
    }
}

fn read_u32_le(bytes: &[u8], pos: usize) -> Result<u32, FormatError> {
    let slice = bytes
        .get(pos..pos.saturating_add(4))
        .ok_or_else(|| FormatError::MalformedMat("Unexpected EOF reading u32".into()))?;
    let arr: [u8; 4] = slice
        .try_into()
        .map_err(|_| FormatError::MalformedMat("Slice conversion failed".into()))?;
    Ok(u32::from_le_bytes(arr))
}

fn read_i32_le(bytes: &[u8], pos: usize) -> Result<i32, FormatError> {
    let slice = bytes
        .get(pos..pos.saturating_add(4))
        .ok_or_else(|| FormatError::MalformedMat("Unexpected EOF reading i32".into()))?;
    let arr: [u8; 4] = slice
        .try_into()
        .map_err(|_| FormatError::MalformedMat("Slice conversion failed".into()))?;
    Ok(i32::from_le_bytes(arr))
}

fn read_f64_le(bytes: &[u8], pos: usize) -> Result<f64, FormatError> {
    let slice = bytes
        .get(pos..pos.saturating_add(8))
        .ok_or_else(|| FormatError::MalformedMat("Unexpected EOF reading f64".into()))?;
    let arr: [u8; 8] = slice
        .try_into()
        .map_err(|_| FormatError::MalformedMat("Slice conversion failed".into()))?;
    Ok(f64::from_le_bytes(arr))
}

/// Writes a numeric `Tensor<f64>` to RFC-4180 CSV text.
pub fn write_csv(tensor: &Tensor<f64>) -> Result<String, FormatError> {
    let mut out = String::new();
    for r in 0..tensor.rows() {
        for c in 0..tensor.cols() {
            if c > 0 {
                out.push(',');
            }
            out.push_str(&tensor.get(r, c)?.to_string());
        }
        out.push('\n');
    }
    Ok(out)
}

/// Reads numeric CSV text into a `Tensor<f64>` with strict budget verification.
pub fn read_csv(text: &str, budget: &Arc<MemoryBudget>) -> Result<Tensor<f64>, FormatError> {
    let mut row_vecs: Vec<Vec<f64>> = Vec::new();
    let mut expected_cols: Option<usize> = None;

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut row = Vec::new();
        for cell in line.split(',') {
            let val: f64 = cell.trim().parse().map_err(|_| FormatError::MalformedCsv {
                line: line_idx.saturating_add(1),
                reason: format!("Invalid float literal '{cell}'"),
            })?;
            row.push(val);
        }
        if let Some(cols) = expected_cols {
            if row.len() != cols {
                return Err(FormatError::MalformedCsv {
                    line: line_idx.saturating_add(1),
                    reason: format!("Expected {cols} columns, found {}", row.len()),
                });
            }
        } else {
            expected_cols = Some(row.len());
        }
        row_vecs.push(row);
    }

    let rows = row_vecs.len();
    let cols = expected_cols.unwrap_or(0);
    let mut col_major = Vec::with_capacity(rows.saturating_mul(cols));
    for c in 0..cols {
        for r in 0..rows {
            let val = row_vecs
                .get(r)
                .and_then(|row| row.get(c))
                .copied()
                .unwrap_or(0.0);
            col_major.push(val);
        }
    }
    Ok(Tensor::from_column_major(
        rows,
        cols,
        col_major,
        SiUnit::DIMENSIONLESS,
        budget,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat5_and_csv_roundtrip_and_panic_hunt() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        let a = Tensor::from_column_major(
            2,
            3,
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            SiUnit::DIMENSIONLESS,
            &budget,
        )?;
        let mut map = BTreeMap::new();
        map.insert("A".to_string(), a.clone());

        let bytes = write_mat5(&map)?;
        let decoded = read_mat5(&bytes, &budget)?;
        let a_back = decoded.get("A").ok_or("Missing A")?;
        assert_eq!(a_back.shape(), [2, 3]);
        assert_eq!(a_back.get(1, 2)?, 6.0);

        // CSV roundtrip
        let csv_str = write_csv(&a)?;
        let csv_back = read_csv(&csv_str, &budget)?;
        assert_eq!(csv_back.get(1, 2)?, 6.0);

        // Hostile / truncated .mat input must return Err, never panic
        assert!(read_mat5(&[0u8; 10], &budget).is_err());
        let mut corrupted = bytes.clone();
        if let Some(b) = corrupted.get_mut(132) {
            *b = 0xFF; // Corrupt miMATRIX byte length to huge value
        }
        assert!(read_mat5(&corrupted, &budget).is_err());
        Ok(())
    }
}
