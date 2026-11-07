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

use engine::{EngineCommand, EngineError, EngineOutput, Session, VarSummary};
use num_core::MemoryBudget;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum UiError {
    #[error("Engine error: {0}")]
    Engine(#[from] EngineError),
    #[error("UI state error: {0}")]
    State(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsoleEntry {
    pub input: String,
    pub output: Result<String, String>,
}

/// Thin UI View-Model for the MatCraft desktop/web interface.
/// Encapsulates command input, history navigation, console output, variable inspection,
/// and figure visualization state.
#[derive(Debug)]
pub struct UiViewModel {
    pub session: Session,
    pub input_buffer: String,
    pub command_history: Vec<String>,
    pub history_cursor: Option<usize>,
    pub console_log: Vec<ConsoleEntry>,
    pub active_figure_id: Option<usize>,
    pub active_figure_svg: Option<String>,
}

impl UiViewModel {
    #[must_use]
    pub fn new(budget: Arc<MemoryBudget>) -> Self {
        Self {
            session: Session::new(budget),
            input_buffer: String::new(),
            command_history: Vec::new(),
            history_cursor: None,
            console_log: Vec::new(),
            active_figure_id: None,
            active_figure_svg: None,
        }
    }

    pub fn submit_input(&mut self) -> Result<String, UiError> {
        let trimmed = self.input_buffer.trim().to_string();
        if trimmed.is_empty() {
            return Ok(String::new());
        }
        self.command_history.push(trimmed.clone());
        self.history_cursor = None;
        self.input_buffer.clear();

        let res = self.session.execute(EngineCommand::EvalM {
            script: trimmed.clone(),
        });
        match res {
            Ok(output) => {
                let msg = match output {
                    EngineOutput::EvalResult { text, ans_summary } => {
                        if let Some(ans) = ans_summary {
                            format!("{text}\nans = {ans}")
                        } else {
                            text
                        }
                    }
                    other => format!("{other:?}"),
                };
                self.console_log.push(ConsoleEntry {
                    input: trimmed,
                    output: Ok(msg.clone()),
                });
                // Update active figure if any figures were produced
                if let Some(&fig_id) = self.session.workspace.figures.keys().last() {
                    self.active_figure_id = Some(fig_id);
                    if let Ok(EngineOutput::Svg(svg)) =
                        self.session.execute(EngineCommand::RenderFigureSvg {
                            figure_id: fig_id,
                            width: 800,
                            height: 600,
                        })
                    {
                        self.active_figure_svg = Some(svg);
                    }
                }
                Ok(msg)
            }
            Err(err) => {
                let err_msg = err.to_string();
                self.console_log.push(ConsoleEntry {
                    input: trimmed,
                    output: Err(err_msg.clone()),
                });
                Err(UiError::Engine(err))
            }
        }
    }

    pub fn undo(&mut self) -> Result<String, UiError> {
        let out = self.session.undo().map_err(UiError::Engine)?;
        let msg = match out {
            EngineOutput::UndoRedoOk {
                undo_remaining,
                redo_remaining,
            } => {
                format!("Undo OK (undo: {undo_remaining}, redo: {redo_remaining})")
            }
            _ => "Undo OK".to_string(),
        };
        Ok(msg)
    }

    pub fn redo(&mut self) -> Result<String, UiError> {
        let out = self.session.redo().map_err(UiError::Engine)?;
        let msg = match out {
            EngineOutput::UndoRedoOk {
                undo_remaining,
                redo_remaining,
            } => {
                format!("Redo OK (undo: {undo_remaining}, redo: {redo_remaining})")
            }
            _ => "Redo OK".to_string(),
        };
        Ok(msg)
    }

    pub fn variables(&mut self) -> Result<Vec<VarSummary>, UiError> {
        let out = self
            .session
            .execute(EngineCommand::ListVariables)
            .map_err(UiError::Engine)?;
        if let EngineOutput::VariableList(vars) = out {
            Ok(vars)
        } else {
            Ok(Vec::new())
        }
    }
}
