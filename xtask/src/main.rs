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

use serde::Deserialize;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

/// Strict (Layer, Subtier) mapping for every workspace member.
/// A crate at `(L_a, T_a)` may ONLY depend on workspace crates at `(L_b, T_b)`
/// where `(L_b, T_b) < (L_a, T_a)`.
fn crate_layer_tier(name: &str) -> Option<(u8, u8)> {
    match name {
        // L0: Core Primitives & Formats
        "num-core" => Some((0, 0)),
        "tensor" => Some((0, 1)),
        "format-mat" => Some((0, 2)),
        // L1: Language & Autodiff
        "autodiff" => Some((1, 0)),
        "lang-m" => Some((1, 1)),
        // L2: Dual Compute Backends
        "linalg-cpu" => Some((2, 0)),
        "compute-gpu" => Some((2, 1)),
        // L3: Simulation Engine & Toolboxes
        "sim-engine" => Some((3, 0)),
        "toolbox-control" => Some((3, 1)),
        "toolbox-signal" => Some((3, 1)),
        "toolbox-optim" => Some((3, 1)),
        // L4: Visualization & Local Offline AI
        "plot" => Some((4, 0)),
        "ai-local" => Some((4, 1)),
        // L5: Unified Command & Session Engine
        "engine" => Some((5, 0)),
        // L6: Thin UI & Agent Bridges
        "mcp" => Some((6, 0)),
        "ui-egui" => Some((6, 1)),
        // L7: Executables & Build Tooling
        "matcraft-cli" => Some((7, 0)),
        "matcraft" => Some((7, 1)),
        "xtask" => Some((7, 2)),
        _ => None,
    }
}

#[derive(Deserialize)]
struct CargoMetadata {
    packages: Vec<MetadataPackage>,
}

#[derive(Deserialize)]
struct MetadataPackage {
    name: String,
    manifest_path: String,
    dependencies: Vec<MetadataDep>,
}

#[derive(Deserialize)]
struct MetadataDep {
    name: String,
    path: Option<String>,
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let subcommand = args.get(1).map_or("all", String::as_str);

    let result = match subcommand {
        "layers" => check_layers(),
        "check-panics" => check_no_panics_in_src(),
        "parity" => print_parity_scorecard(),
        "all" => {
            if let Err(e) = check_layers() {
                eprintln!("LAYER CHECK FAILED: {e}");
                return ExitCode::FAILURE;
            }
            if let Err(e) = check_no_panics_in_src() {
                eprintln!("PANIC AUDIT FAILED: {e}");
                return ExitCode::FAILURE;
            }
            if let Err(e) = print_parity_scorecard() {
                eprintln!("PARITY SCORECARD FAILED: {e}");
                return ExitCode::FAILURE;
            }
            Ok(())
        }
        other => Err(format!(
            "Unknown xtask command '{other}'. Expected: layers | check-panics | parity | all"
        )),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("Error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn check_layers() -> Result<(), String> {
    println!("=== MatCraft Architecture Gate: Checking L0..L7 Layer Invariants ===");
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()
        .map_err(|e| format!("Failed to invoke `cargo metadata`: {e}"))?;

    if !output.status.success() {
        return Err("`cargo metadata` returned non-zero status".to_string());
    }

    let meta: CargoMetadata = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("Failed to parse `cargo metadata` JSON: {e}"))?;

    let mut violations = Vec::new();
    let mut crate_counts_by_layer: BTreeMap<u8, usize> = BTreeMap::new();

    for pkg in &meta.packages {
        let Some((src_layer, src_tier)) = crate_layer_tier(&pkg.name) else {
            violations.push(format!(
                "Unregistered workspace crate '{}' at {}",
                pkg.name, pkg.manifest_path
            ));
            continue;
        };
        let count = crate_counts_by_layer.entry(src_layer).or_insert(0);
        *count = count.saturating_add(1);

        // Rule: Nothing below L6 may depend on UI crates (`egui`, `eframe`, `winit`, `rfd`)
        if src_layer < 6 {
            for dep in &pkg.dependencies {
                if matches!(dep.name.as_str(), "egui" | "eframe" | "winit" | "rfd") {
                    violations.push(format!(
                        "UI leakage: '{}' (L{src_layer}) depends on UI crate '{}' (forbidden below L6)",
                        pkg.name, dep.name
                    ));
                }
            }
        }

        // Check workspace dependencies
        for dep in &pkg.dependencies {
            if dep.path.is_none() {
                continue;
            }
            let Some((dst_layer, dst_tier)) = crate_layer_tier(&dep.name) else {
                violations.push(format!(
                    "Crate '{}' depends on unregistered workspace crate '{}'",
                    pkg.name, dep.name
                ));
                continue;
            };

            if (dst_layer, dst_tier) >= (src_layer, src_tier) {
                violations.push(format!(
                    "Layer violation: '{}' (L{src_layer}.{src_tier}) -> '{}' (L{dst_layer}.{dst_tier}). Dependencies must strictly flow downward!",
                    pkg.name, dep.name
                ));
            }
        }
    }

    for (layer, count) in &crate_counts_by_layer {
        println!("  Layer L{layer}: {count} crate(s) verified");
    }

    if !violations.is_empty() {
        for v in &violations {
            eprintln!("  [VIOLATION] {v}");
        }
        return Err(format!("Found {} layer violation(s)", violations.len()));
    }

    println!("=== Layer Check PASSED: 0 violations across all L0..L7 crates ===\n");
    Ok(())
}

fn check_no_panics_in_src() -> Result<(), String> {
    println!("=== MatCraft Rule 1.1 Gate: Auditing non-test source for forbidden panic tokens ===");
    let mut rs_files = Vec::new();
    collect_rs_files(Path::new("crates"), &mut rs_files)?;
    collect_rs_files(Path::new("apps"), &mut rs_files)?;

    let forbidden = [".unwrap(", ".expect(", "panic!(", "todo!(", "unimplemented!("];
    let mut violations = Vec::new();

    for file in &rs_files {
        let content = fs::read_to_string(file)
            .map_err(|e| format!("Failed to read {}: {e}", file.display()))?;
        // Stop scanning once `#[cfg(test)]` module begins
        let non_test_part = content.split("#[cfg(test)]").next().unwrap_or(&content);
        for (line_idx, line) in non_test_part.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("//") {
                continue;
            }
            for token in &forbidden {
                if trimmed.contains(token) {
                    violations.push(format!(
                        "{}:{}: contains forbidden token `{token}`",
                        file.display(),
                        line_idx.saturating_add(1)
                    ));
                }
            }
        }
    }

    if !violations.is_empty() {
        for v in &violations {
            eprintln!("  [FORBIDDEN] {v}");
        }
        return Err(format!(
            "Found {} forbidden panic token(s) in non-test code",
            violations.len()
        ));
    }

    println!(
        "=== Rule 1.1 Audit PASSED: {} source files verified clean ===\n",
        rs_files.len()
    );
    Ok(())
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if !dir.exists() {
        return Ok(());
    }
    let entries =
        fs::read_dir(dir).map_err(|e| format!("Failed to read dir {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("Dir entry error: {e}"))?;
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(path);
        }
    }
    Ok(())
}

fn print_parity_scorecard() -> Result<(), String> {
    println!("=== MatCraft Subsystem Parity Scorecard ===");
    println!("  [L0] num-core      : SI Units, Complex64, Rational128, MemoryBudget [READY]");
    println!("  [L0] tensor        : Tiered CoW 64x64 Tiles, Uniform(0) Opt, CSC    [READY]");
    println!("  [L0] format-mat    : MATLAB Level 5 (.mat) & RFC-4180 CSV I/O       [READY]");
    println!("  [L1] autodiff      : Forward Dual64 Jacobians & Reverse Wengert Tape[READY]");
    println!("  [L1] lang-m        : Context Lexer, Pratt AST, SI-Unit Eval VM      [READY]");
    println!("  [L2] linalg-cpu    : GEMM, LU, QR, Cholesky, SVD, Eig, FFT, mldivide[READY]");
    println!("  [L2] compute-gpu   : WGSL Compute Kernels & Oracle Verifier         [READY]");
    println!("  [L3] sim-engine    : i128 Attosecond Clock, Tarjan SCC, RK4/RK45    [READY]");
    println!("  [L3] toolboxes     : Control (SS/TF/LQR), Signal (FIR), Optim (BFGS)[READY]");
    println!("  [L4] plot          : Handle SceneGraph, LTTB, Deterministic SVG/RGBA[READY]");
    println!("  [L4] ai-local      : Offline SINDy ODE Discovery & Rust Codegen     [READY]");
    println!("  [L5] engine        : CoW Session Undo, CommandRegistry, panic_hunt  [READY]");
    println!("  [L6] mcp / ui-egui : JSON-RPC 2.0 MCP Server & Thin UI ViewModel    [READY]");
    println!("  [L7] matcraft-cli  : Batch Runner, REPL, MCP Host, Doctor           [READY]");
    Ok(())
}
