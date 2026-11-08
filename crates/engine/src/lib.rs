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

pub mod workspace_io;
pub use workspace_io::{load_workspace, save_workspace};

use ai_local::{sindy_discover_1d, transpile_m_expr_to_no_std_rust, AiLocalError};
use format_mat::{read_csv, read_mat5, write_csv, write_mat5, FormatError};
use lang_m::{ExternalBuiltinProvider, Interpreter, LangError};
use linalg_cpu::{chol, det, inv, lu, matmul, mldivide, qr, svd, LinalgError};
use num_core::{AllocError, MemoryBudget, SiUnit, UnitError};
use plot::{FigureScene, LineSeries2D};
use serde::{Deserialize, Serialize};
use sim_engine::{SimError, SimModel};
use std::collections::BTreeMap;
use std::sync::Arc;
use tensor::{Tensor, TensorError};
use thiserror::Error;
use toolbox_signal::fir1_lowpass;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum EngineError {
    #[error("Unknown command ID `{0}`")]
    UnknownCommand(String),
    #[error("Command `{id}` is currently disabled: {reason}")]
    CommandDisabled { id: String, reason: String },
    #[error("Invalid parameter `{param}`: {reason}")]
    InvalidParam { param: &'static str, reason: String },
    #[error("Allocation error: {0}")]
    Alloc(#[from] AllocError),
    #[error("Unit error: {0}")]
    Unit(#[from] UnitError),
    #[error("Tensor error: {0}")]
    Tensor(#[from] TensorError),
    #[error("Language error: {0}")]
    Lang(#[from] LangError),
    #[error("Linear algebra error: {0}")]
    Linalg(#[from] LinalgError),
    #[error("Simulation error: {0}")]
    Sim(#[from] SimError),
    #[error("Format error: {0}")]
    Format(#[from] FormatError),
    #[error("Local AI error: {0}")]
    Ai(#[from] AiLocalError),
    #[error("Plot error: {0}")]
    Plot(#[from] plot::PlotError),
    #[error("Signal toolbox error: {0}")]
    Signal(#[from] toolbox_signal::SignalError),
    #[error("Nothing to undo in workspace history")]
    EmptyUndoStack,
    #[error("Nothing to redo in workspace history")]
    EmptyRedoStack,
    #[error("Variable '{0}' not found in workspace")]
    VariableNotFound(String),
    #[error("Figure '{0}' not found in workspace")]
    FigureNotFound(usize),
}

/// Copy-on-Write Workspace State.
/// Cloning `Workspace` is O(1) because every `Tensor<f64>` is `Arc`-backed!
#[derive(Clone, Debug)]
pub struct Workspace {
    pub vars: BTreeMap<String, Tensor<f64>>,
    pub figures: BTreeMap<usize, FigureScene>,
    pub sim_model: SimModel,
    pub budget: Arc<MemoryBudget>,
}

impl Workspace {
    #[must_use]
    pub fn new(budget: Arc<MemoryBudget>) -> Self {
        Self {
            vars: BTreeMap::new(),
            figures: BTreeMap::new(),
            sim_model: SimModel::new(),
            budget,
        }
    }
}

#[derive(Debug)]
pub struct EngineExternalProvider<'a> {
    pub figures: &'a mut BTreeMap<usize, FigureScene>,
    pub next_fig_id: &'a mut usize,
}

impl<'a> ExternalBuiltinProvider for EngineExternalProvider<'a> {
    fn call_external(
        &mut self,
        name: &str,
        args: &[Tensor<f64>],
        budget: &Arc<MemoryBudget>,
    ) -> Result<Option<Vec<Tensor<f64>>>, LangError> {
        match name {
            "matmul" => {
                let a = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "matmul".into(),
                    reason: "Missing left matrix".into(),
                })?;
                let b = args.get(1).ok_or_else(|| LangError::BuiltinError {
                    name: "matmul".into(),
                    reason: "Missing right matrix".into(),
                })?;
                let res = matmul(a, b, budget).map_err(|e| LangError::BuiltinError {
                    name: "matmul".into(),
                    reason: e.to_string(),
                })?;
                Ok(Some(vec![res]))
            }
            "mldivide" => {
                let a = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "mldivide".into(),
                    reason: "Missing left operand".into(),
                })?;
                let b = args.get(1).ok_or_else(|| LangError::BuiltinError {
                    name: "mldivide".into(),
                    reason: "Missing right operand".into(),
                })?;
                let res = mldivide(a, b, budget).map_err(|e| LangError::BuiltinError {
                    name: "mldivide".into(),
                    reason: e.to_string(),
                })?;
                Ok(Some(vec![res]))
            }
            "inv" => {
                let a = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "inv".into(),
                    reason: "Missing matrix argument".into(),
                })?;
                let res = inv(a, budget).map_err(|e| LangError::BuiltinError {
                    name: "inv".into(),
                    reason: e.to_string(),
                })?;
                Ok(Some(vec![res]))
            }
            "det" => {
                let a = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "det".into(),
                    reason: "Missing matrix argument".into(),
                })?;
                let val = det(a, budget).map_err(|e| LangError::BuiltinError {
                    name: "det".into(),
                    reason: e.to_string(),
                })?;
                let t = Tensor::scalar(val, SiUnit::DIMENSIONLESS, budget)
                    .map_err(LangError::Tensor)?;
                Ok(Some(vec![t]))
            }
            "lu" => {
                let a = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "lu".into(),
                    reason: "Missing matrix argument".into(),
                })?;
                let res = lu(a, budget).map_err(|e| LangError::BuiltinError {
                    name: "lu".into(),
                    reason: e.to_string(),
                })?;
                Ok(Some(vec![res.l, res.u, res.p]))
            }
            "qr" => {
                let a = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "qr".into(),
                    reason: "Missing matrix argument".into(),
                })?;
                let (q, r) = qr(a, budget).map_err(|e| LangError::BuiltinError {
                    name: "qr".into(),
                    reason: e.to_string(),
                })?;
                Ok(Some(vec![q, r]))
            }
            "chol" => {
                let a = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "chol".into(),
                    reason: "Missing matrix argument".into(),
                })?;
                let res = chol(a, budget).map_err(|e| LangError::BuiltinError {
                    name: "chol".into(),
                    reason: e.to_string(),
                })?;
                Ok(Some(vec![res]))
            }
            "svd" => {
                let a = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "svd".into(),
                    reason: "Missing matrix argument".into(),
                })?;
                let (u, s, v) = svd(a, budget).map_err(|e| LangError::BuiltinError {
                    name: "svd".into(),
                    reason: e.to_string(),
                })?;
                Ok(Some(vec![u, s, v]))
            }
            "fir1" => {
                let order_t = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "fir1".into(),
                    reason: "Missing order argument".into(),
                })?;
                let cutoff_t = args.get(1).ok_or_else(|| LangError::BuiltinError {
                    name: "fir1".into(),
                    reason: "Missing cutoff argument".into(),
                })?;
                let order = order_t.get(0, 0).map_err(LangError::Tensor)? as usize;
                let cutoff = cutoff_t.get(0, 0).map_err(LangError::Tensor)?;
                let taps = fir1_lowpass(order, cutoff);
                let res = Tensor::from_column_major(
                    1,
                    taps.len(),
                    taps,
                    SiUnit::DIMENSIONLESS,
                    budget,
                )
                .map_err(LangError::Tensor)?;
                Ok(Some(vec![res]))
            }
            "plot" => {
                let id = *self.next_fig_id;
                *self.next_fig_id = self.next_fig_id.saturating_add(1);
                let mut scene = FigureScene::new(id, "Figure");

                if args.len() == 1 {
                    let y_t = args.first().ok_or_else(|| LangError::BuiltinError {
                        name: "plot".into(),
                        reason: "Missing Y tensor".into(),
                    })?;
                    let y_vals = y_t.to_column_major_vec().map_err(LangError::Tensor)?;
                    let x_vals: Vec<f64> = (1..=y_vals.len()).map(|i| i as f64).collect();
                    scene.series.push(LineSeries2D {
                        label: "y".to_string(),
                        x: x_vals,
                        y: y_vals,
                        color_rgb: [56, 139, 253],
                    });
                } else if args.len() >= 2 {
                    let x_t = args.first().ok_or_else(|| LangError::BuiltinError {
                        name: "plot".into(),
                        reason: "Missing X tensor".into(),
                    })?;
                    let y_t = args.get(1).ok_or_else(|| LangError::BuiltinError {
                        name: "plot".into(),
                        reason: "Missing Y tensor".into(),
                    })?;
                    let x_vals = x_t.to_column_major_vec().map_err(LangError::Tensor)?;
                    let y_vals = y_t.to_column_major_vec().map_err(LangError::Tensor)?;
                    scene.series.push(LineSeries2D {
                        label: "trace".to_string(),
                        x: x_vals,
                        y: y_vals,
                        color_rgb: [56, 139, 253],
                    });
                }
                self.figures.insert(id, scene);
                let res = Tensor::scalar(id as f64, SiUnit::DIMENSIONLESS, budget)
                    .map_err(LangError::Tensor)?;
                Ok(Some(vec![res]))
            }
            _ => Ok(None),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VarSummary {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub unit: String,
    pub sample_val: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EngineCommand {
    EvalM { script: String },
    GetVariable { name: String },
    SetVariable {
        name: String,
        values: Vec<f64>,
        rows: usize,
        cols: usize,
        unit: String,
    },
    ListVariables,
    RenderFigureSvg { figure_id: usize, width: u32, height: u32 },
    ExportMat5,
    ImportMat5 { bytes: Vec<u8> },
    ExportCsv { variable_name: String },
    ImportCsv { variable_name: String, csv_text: String },
    RunSimulation { duration_sec: f64, dt_sec: f64 },
    DiscoverOdeSindy {
        x_samples: Vec<f64>,
        dx_samples: Vec<f64>,
        threshold: f64,
    },
    TranspileRust { fn_name: String, expr: String },
    PanicHuntFuzz { iterations: usize },
    Undo,
    Redo,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EngineOutput {
    EvalResult {
        text: String,
        ans_summary: Option<String>,
    },
    VariableData {
        name: String,
        rows: usize,
        cols: usize,
        unit: String,
        values: Vec<f64>,
    },
    VariableList(Vec<VarSummary>),
    Svg(String),
    Mat5Bytes(Vec<u8>),
    CsvText(String),
    SimResult {
        times_count: usize,
        signal_names: Vec<String>,
    },
    DiscoveredEquation {
        equation: String,
        coefficients: Vec<f64>,
    },
    RustCode(String),
    FuzzAuditPassed {
        iterations: usize,
        errors_caught_safely: usize,
    },
    UndoRedoOk {
        undo_remaining: usize,
        redo_remaining: usize,
    },
}

#[derive(Debug)]
pub struct Session {
    pub workspace: Workspace,
    pub undo_stack: Vec<Workspace>,
    pub redo_stack: Vec<Workspace>,
    pub max_undo_depth: usize,
    pub next_fig_id: usize,
}

impl Session {
    #[must_use]
    pub fn new(budget: Arc<MemoryBudget>) -> Self {
        Self {
            workspace: Workspace::new(budget),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_undo_depth: 64,
            next_fig_id: 1,
        }
    }

    fn push_undo(&mut self) {
        if self.max_undo_depth == 0 {
            return;
        }
        if self.undo_stack.len() >= self.max_undo_depth && !self.undo_stack.is_empty() {
            let _ = self.undo_stack.drain(0..1);
        }
        self.undo_stack.push(self.workspace.clone());
        self.redo_stack.clear();
    }

    pub fn undo(&mut self) -> Result<EngineOutput, EngineError> {
        let prev = self.undo_stack.pop().ok_or(EngineError::EmptyUndoStack)?;
        self.redo_stack.push(self.workspace.clone());
        self.workspace = prev;
        Ok(EngineOutput::UndoRedoOk {
            undo_remaining: self.undo_stack.len(),
            redo_remaining: self.redo_stack.len(),
        })
    }

    pub fn redo(&mut self) -> Result<EngineOutput, EngineError> {
        let next = self.redo_stack.pop().ok_or(EngineError::EmptyRedoStack)?;
        self.undo_stack.push(self.workspace.clone());
        self.workspace = next;
        Ok(EngineOutput::UndoRedoOk {
            undo_remaining: self.undo_stack.len(),
            redo_remaining: self.redo_stack.len(),
        })
    }

    pub fn execute(&mut self, cmd: EngineCommand) -> Result<EngineOutput, EngineError> {
        match cmd {
            EngineCommand::Undo => self.undo(),
            EngineCommand::Redo => self.redo(),
            EngineCommand::EvalM { script } => {
                self.push_undo();
                let mut interp = Interpreter::new(Arc::clone(&self.workspace.budget));
                interp.vars = self.workspace.vars.clone();

                let mut provider = EngineExternalProvider {
                    figures: &mut self.workspace.figures,
                    next_fig_id: &mut self.next_fig_id,
                };

                let res = interp.eval_script(&script, &mut provider)?;
                self.workspace.vars = interp.vars;

                let ans_summary = if let Some(t) = res {
                    let s = format!(
                        "[{rows}x{cols} {unit}] = {val}",
                        rows = t.rows(),
                        cols = t.cols(),
                        unit = t.unit(),
                        val = t.get(0, 0).map_or_else(|_| "NaN".to_string(), |v| v.to_string())
                    );
                    Some(s)
                } else {
                    None
                };

                Ok(EngineOutput::EvalResult {
                    text: "Execution succeeded".to_string(),
                    ans_summary,
                })
            }
            EngineCommand::GetVariable { name } => {
                let t = self
                    .workspace
                    .vars
                    .get(&name)
                    .ok_or_else(|| EngineError::VariableNotFound(name.clone()))?;
                let values = t.to_column_major_vec()?;
                Ok(EngineOutput::VariableData {
                    name,
                    rows: t.rows(),
                    cols: t.cols(),
                    unit: t.unit().to_string(),
                    values,
                })
            }
            EngineCommand::SetVariable {
                name,
                values,
                rows,
                cols,
                unit,
            } => {
                self.push_undo();
                let si = SiUnit::parse(&unit)?;
                let t = Tensor::from_column_major(
                    rows,
                    cols,
                    values,
                    si,
                    &self.workspace.budget,
                )?;
                self.workspace.vars.insert(name, t);
                Ok(EngineOutput::EvalResult {
                    text: "Variable updated".to_string(),
                    ans_summary: None,
                })
            }
            EngineCommand::ListVariables => {
                let mut list = Vec::with_capacity(self.workspace.vars.len());
                for (name, t) in &self.workspace.vars {
                    list.push(VarSummary {
                        name: name.clone(),
                        rows: t.rows(),
                        cols: t.cols(),
                        unit: t.unit().to_string(),
                        sample_val: t.get(0, 0).ok(),
                    });
                }
                Ok(EngineOutput::VariableList(list))
            }
            EngineCommand::RenderFigureSvg {
                figure_id,
                width,
                height,
            } => {
                let fig = self
                    .workspace
                    .figures
                    .get(&figure_id)
                    .ok_or(EngineError::FigureNotFound(figure_id))?;
                let svg = fig.render_to_svg(width, height);
                Ok(EngineOutput::Svg(svg))
            }
            EngineCommand::ExportMat5 => {
                let bytes = write_mat5(&self.workspace.vars)?;
                Ok(EngineOutput::Mat5Bytes(bytes))
            }
            EngineCommand::ImportMat5 { bytes } => {
                self.push_undo();
                let imported = read_mat5(&bytes, &self.workspace.budget)?;
                for (k, v) in imported {
                    self.workspace.vars.insert(k, v);
                }
                Ok(EngineOutput::EvalResult {
                    text: "Imported .mat file variables into workspace".to_string(),
                    ans_summary: None,
                })
            }
            EngineCommand::ExportCsv { variable_name } => {
                let t = self
                    .workspace
                    .vars
                    .get(&variable_name)
                    .ok_or_else(|| EngineError::VariableNotFound(variable_name.clone()))?;
                let csv = write_csv(t)?;
                Ok(EngineOutput::CsvText(csv))
            }
            EngineCommand::ImportCsv {
                variable_name,
                csv_text,
            } => {
                self.push_undo();
                let t = read_csv(&csv_text, &self.workspace.budget)?;
                self.workspace.vars.insert(variable_name, t);
                Ok(EngineOutput::EvalResult {
                    text: "Imported CSV variable into workspace".to_string(),
                    ans_summary: None,
                })
            }
            EngineCommand::RunSimulation {
                duration_sec,
                dt_sec,
            } => {
                let trace = self.workspace.sim_model.simulate_rk4(duration_sec, dt_sec)?;
                let signal_names: Vec<String> = trace.signals.keys().cloned().collect();
                Ok(EngineOutput::SimResult {
                    times_count: trace.times_sec.len(),
                    signal_names,
                })
            }
            EngineCommand::DiscoverOdeSindy {
                x_samples,
                dx_samples,
                threshold,
            } => {
                let ode = sindy_discover_1d(
                    &x_samples,
                    &dx_samples,
                    threshold,
                    &self.workspace.budget,
                )?;
                Ok(EngineOutput::DiscoveredEquation {
                    equation: ode.formatted_equation,
                    coefficients: ode.coefficients,
                })
            }
            EngineCommand::TranspileRust { fn_name, expr } => {
                let code = transpile_m_expr_to_no_std_rust(&fn_name, &expr);
                Ok(EngineOutput::RustCode(code))
            }
            EngineCommand::PanicHuntFuzz { iterations } => {
                let errors_caught = run_panic_hunt(iterations, &self.workspace.budget)?;
                Ok(EngineOutput::FuzzAuditPassed {
                    iterations,
                    errors_caught_safely: errors_caught,
                })
            }
        }
    }
}

/// Panic Hunt Audit: Systematically exercises untrusted, hostile, or malformed inputs
/// verifying that zero panics occur and that errors are returned safely.
pub fn run_panic_hunt(
    iterations: usize,
    budget: &Arc<MemoryBudget>,
) -> Result<usize, EngineError> {
    let mut errors_caught = 0usize;
    let iters = iterations.clamp(1, 10_000);

    for i in 0..iters {
        // 1. Hostile / malformed script strings
        let bad_code = match i % 6 {
            0 => "A = [1, 2; 3]",             // Mismatched matrix row lengths
            1 => "x = 1/0; y = tan(x)",        // Division by zero & transcendental
            2 => "for = 1:5; end",             // Syntax error in for loop
            3 => "m = 5_kg; a = 2_s; F = m + a", // Incompatible physical units
            4 => "v = zeros(2, 2); v(5, 5)",   // Out of bounds indexing
            _ => "A = zeros(3, 3); inv(A)",    // Singular matrix inversion
        };

        let mut interp = Interpreter::new(Arc::clone(budget));
        let mut ext = lang_m::NoopExternalProvider;
        if interp.eval_script(bad_code, &mut ext).is_err() {
            errors_caught = errors_caught.saturating_add(1);
        }

        // 2. Hostile / truncated .mat binary data
        let mut bogus_bytes = vec![0u8; (i % 64).saturating_add(16)];
        if let Some(b) = bogus_bytes.get_mut(0) {
            *b = (i as u8).wrapping_mul(37);
        }
        if read_mat5(&bogus_bytes, budget).is_err() {
            errors_caught = errors_caught.saturating_add(1);
        }

        // 3. Hostile CSV parsing
        let bogus_csv = "1.0, 2.0\n3.0, not_a_number\n";
        if read_csv(bogus_csv, budget).is_err() {
            errors_caught = errors_caught.saturating_add(1);
        }
    }

    Ok(errors_caught)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_undo_redo_and_external_builtins() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        let mut session = Session::new(budget);

        // 1. Execute script with linear solve and unit propagation
        let cmd = EngineCommand::EvalM {
            script: "A = [4, 1; 2, 3]; b = [1; 1]; x = A \\ b".to_string(),
        };
        let out = session.execute(cmd)?;
        assert!(matches!(out, EngineOutput::EvalResult { .. }));

        let var_out = session.execute(EngineCommand::GetVariable {
            name: "x".to_string(),
        })?;
        if let EngineOutput::VariableData { values, .. } = var_out {
            let v0 = values.first().copied().unwrap_or(0.0);
            assert!((v0 - 0.2).abs() < 1e-10);
        } else {
            return Err("Expected VariableData".into());
        }

        // 2. Verify Undo / Redo
        session.undo()?;
        assert!(session
            .execute(EngineCommand::GetVariable {
                name: "x".to_string()
            })
            .is_err());
        session.redo()?;
        assert!(session
            .execute(EngineCommand::GetVariable {
                name: "x".to_string()
            })
            .is_ok());

        // 3. Verify Panic Hunt audit
        let hunt = session.execute(EngineCommand::PanicHuntFuzz { iterations: 50 })?;
        if let EngineOutput::FuzzAuditPassed {
            iterations,
            errors_caught_safely,
        } = hunt
        {
            assert_eq!(iterations, 50);
            assert!(errors_caught_safely > 0);
        } else {
            return Err("Expected FuzzAuditPassed".into());
        }

        Ok(())
    }
}
