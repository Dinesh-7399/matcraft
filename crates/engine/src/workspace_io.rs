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

//! Workspace snapshot save/load via JSON serialization.

use crate::EngineError;
use num_core::{MemoryBudget, SiUnit};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use tensor::Tensor;

/// JSON representation of a single variable in the workspace snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct VariableEntry {
    name: String,
    rows: usize,
    cols: usize,
    unit: SiUnit,
    data: Vec<f64>,
}

/// Save workspace variables to a JSON file at `path`.
pub fn save_workspace(
    vars: &BTreeMap<String, Tensor<f64>>,
    path: &Path,
) -> Result<(), EngineError> {
    let mut entries: Vec<VariableEntry> = Vec::with_capacity(vars.len());

    for (name, tensor) in vars {
        let [rows, cols] = tensor.shape();
        let data = tensor
            .to_column_major_vec()
            .map_err(EngineError::Tensor)?;

        entries.push(VariableEntry {
            name: name.clone(),
            rows,
            cols,
            unit: tensor.unit(),
            data,
        });
    }

    let json = serde_json::to_string_pretty(&entries)
        .map_err(|e| EngineError::InvalidParam {
            param: "workspace_io::save",
            reason: e.to_string(),
        })?;

    std::fs::write(path, json).map_err(|e| EngineError::InvalidParam {
        param: "workspace_io::save (write)",
        reason: e.to_string(),
    })?;

    Ok(())
}

/// Load workspace variables from a JSON file at `path`.
///
/// Returns a `BTreeMap<String, Tensor<f64>>` reconstructed from the JSON snapshot.
pub fn load_workspace(path: &Path) -> Result<BTreeMap<String, Tensor<f64>>, EngineError> {
    let content = std::fs::read_to_string(path).map_err(|e| EngineError::InvalidParam {
        param: "workspace_io::load (read)",
        reason: e.to_string(),
    })?;

    let entries: Vec<VariableEntry> = serde_json::from_str(&content).map_err(|e| {
        EngineError::InvalidParam {
            param: "workspace_io::load (parse)",
            reason: e.to_string(),
        }
    })?;

    let budget = MemoryBudget::default_budget();
    let mut vars = BTreeMap::new();

    for entry in entries {
        let tensor = Tensor::from_column_major(
            entry.rows,
            entry.cols,
            entry.data,
            entry.unit,
            &budget,
        )
        .map_err(EngineError::Tensor)?;

        vars.insert(entry.name, tensor);
    }

    Ok(vars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_load_workspace_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        let mut vars = BTreeMap::new();

        let a = Tensor::from_column_major(
            2, 3,
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            SiUnit::DIMENSIONLESS,
            &budget,
        )?;
        vars.insert("A".to_string(), a);

        let v = Tensor::from_column_major(
            3, 1,
            vec![7.0, 8.0, 9.0],
            SiUnit::DIMENSIONLESS,
            &budget,
        )?;
        vars.insert("v".to_string(), v);

        // Use a temp file
        let tmp = std::env::temp_dir().join("matcraft_test_workspace.json");
        save_workspace(&vars, &tmp)?;

        let loaded = load_workspace(&tmp)?;
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(loaded.len(), 2);
        let a_back = loaded.get("A").ok_or("Missing A")?;
        assert_eq!(a_back.shape(), [2, 3]);
        assert!((a_back.get(1, 2)? - 6.0).abs() < 1e-12);

        let v_back = loaded.get("v").ok_or("Missing v")?;
        assert_eq!(v_back.shape(), [3, 1]);
        assert!((v_back.get(2, 0)? - 9.0).abs() < 1e-12);

        Ok(())
    }

    #[test]
    fn test_load_nonexistent_path_returns_error() {
        let result = load_workspace(Path::new("/nonexistent/path/workspace.json"));
        assert!(result.is_err());
    }
}
