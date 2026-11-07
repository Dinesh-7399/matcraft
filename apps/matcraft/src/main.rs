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

use num_core::MemoryBudget;
use std::process::ExitCode;
use ui_egui::UiViewModel;

fn main() -> ExitCode {
    println!("=== MatCraft Desktop Core ===");
    println!("Initializing MatCraft pure-Rust engine with Golden Rule invariants...");

    let budget = MemoryBudget::default_budget();
    let mut vm = UiViewModel::new(budget);

    // Run demonstration workload validating L0..L6 integration
    vm.input_buffer = "m = 10_kg; a = 9.81_m/s^2; F = m * a; work = F * 5_m".to_string();
    if let Err(e) = vm.submit_input() {
        eprintln!("Failed demo physics step: {e}");
        return ExitCode::FAILURE;
    }

    vm.input_buffer = "A = [4, 1; 2, 3]; b = [1; 1]; x = A \\ b".to_string();
    if let Err(e) = vm.submit_input() {
        eprintln!("Failed demo linear algebra step: {e}");
        return ExitCode::FAILURE;
    }

    println!("Demo execution completed successfully!");
    match vm.variables() {
        Ok(vars) => {
            println!("Active Workspace Variables ({} items):", vars.len());
            for v in vars {
                println!("  - {}: {}x{} [{}]", v.name, v.rows, v.cols, v.unit);
            }
        }
        Err(e) => {
            eprintln!("Failed to list variables: {e}");
            return ExitCode::FAILURE;
        }
    }

    println!("=== MatCraft Desktop Core Ready ===");
    ExitCode::SUCCESS
}
