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

use engine::{EngineCommand, EngineError, EngineOutput, Session};
use num_core::MemoryBudget;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum McpError {
    #[error("JSON error: {0}")]
    Json(String),
    #[error("Engine error: {0}")]
    Engine(#[from] EngineError),
    #[error("Method not found: {0}")]
    MethodNotFound(String),
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpResponseError>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct McpResponseError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Debug)]
pub struct McpServer {
    pub session: Session,
}

impl McpServer {
    #[must_use]
    pub fn new(budget: Arc<MemoryBudget>) -> Self {
        Self {
            session: Session::new(budget),
        }
    }

    pub fn handle_raw(&mut self, request_str: &str) -> String {
        let req: McpRequest = match serde_json::from_str(request_str) {
            Ok(r) => r,
            Err(e) => {
                let err_resp = McpResponse {
                    jsonrpc: "2.0".to_string(),
                    id: None,
                    result: None,
                    error: Some(McpResponseError {
                        code: -32700,
                        message: format!("Parse error: {e}"),
                        data: None,
                    }),
                };
                return serde_json::to_string(&err_resp).unwrap_or_else(|_| "{}".to_string());
            }
        };

        let resp = self.handle_request(req);
        serde_json::to_string(&resp).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn handle_request(&mut self, req: McpRequest) -> McpResponse {
        let id = req.id.clone();
        match self.dispatch(req) {
            Ok(val) => McpResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: Some(val),
                error: None,
            },
            Err(err) => McpResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: None,
                error: Some(McpResponseError {
                    code: -32603,
                    message: err.to_string(),
                    data: None,
                }),
            },
        }
    }

    #[allow(clippy::disallowed_methods)]
    fn dispatch(&mut self, req: McpRequest) -> Result<Value, McpError> {
        match req.method.as_str() {
            "initialize" => Ok(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "listChanged": false
                    }
                },
                "serverInfo": {
                    "name": "matcraft-mcp",
                    "version": "0.1.0"
                }
            })),
            "notifications/initialized" => Ok(json!({})),
            "tools/list" => Ok(json!({
                "tools": [
                    {
                        "name": "eval_m",
                        "description": "Evaluates a MATLAB-compatible .m script in MatCraft pure-Rust engine with physical SI units and linear algebra solvers.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "script": { "type": "string", "description": "The MATLAB script to evaluate" }
                            },
                            "required": ["script"]
                        }
                    },
                    {
                        "name": "list_variables",
                        "description": "Lists all variables currently stored in the workspace symbol table with dimensions and SI units.",
                        "inputSchema": { "type": "object", "properties": {} }
                    },
                    {
                        "name": "get_variable",
                        "description": "Retrieves the values, shape, and SI unit of a workspace variable.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string", "description": "Variable name" }
                            },
                            "required": ["name"]
                        }
                    },
                    {
                        "name": "render_figure",
                        "description": "Renders an interactive scene-graph figure to standalone SVG.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "figure_id": { "type": "integer", "description": "Figure ID" },
                                "width": { "type": "integer", "description": "Width in pixels", "default": 800 },
                                "height": { "type": "integer", "description": "Height in pixels", "default": 600 }
                            },
                            "required": ["figure_id"]
                        }
                    },
                    {
                        "name": "run_simulation",
                        "description": "Simulates continuous/discrete block diagrams with RK4 integration on an attosecond clock grid.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "duration_sec": { "type": "number", "description": "Duration in seconds" },
                                "dt_sec": { "type": "number", "description": "Timestep in seconds", "default": 0.01 }
                            },
                            "required": ["duration_sec"]
                        }
                    },
                    {
                        "name": "discover_dynamics",
                        "description": "Uses offline SINDy to discover closed-form differential equations from time-series state and derivative data.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "x_samples": { "type": "array", "items": { "type": "number" } },
                                "dx_samples": { "type": "array", "items": { "type": "number" } },
                                "threshold": { "type": "number", "default": 0.001 }
                            },
                            "required": ["x_samples", "dx_samples"]
                        }
                    },
                    {
                        "name": "panic_hunt",
                        "description": "Executes the Rule 1.1 fuzz audit over hostile, truncated, and corrupted inputs to verify 0 panics.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "iterations": { "type": "integer", "default": 100 }
                            }
                        }
                    }
                ]
            })),
            "tools/call" => {
                let tool_name = req.params.get("name").and_then(Value::as_str).ok_or_else(|| {
                    McpError::InvalidRequest("Missing tool `name` in call parameters".into())
                })?;
                let arguments = req.params.get("arguments").cloned().unwrap_or(json!({}));
                self.call_tool(tool_name, arguments)
            }
            other => Err(McpError::MethodNotFound(other.to_string())),
        }
    }

    #[allow(clippy::disallowed_methods)]
    fn call_tool(&mut self, name: &str, args: Value) -> Result<Value, McpError> {
        match name {
            "eval_m" => {
                let script = args.get("script").and_then(Value::as_str).ok_or_else(|| {
                    McpError::InvalidRequest("Missing `script` argument".into())
                })?;
                let out = self.session.execute(EngineCommand::EvalM {
                    script: script.to_string(),
                })?;
                if let EngineOutput::EvalResult { text, ans_summary } = out {
                    let summary_str = ans_summary.unwrap_or_else(|| "Done".to_string());
                    Ok(json!({
                        "content": [
                            {
                                "type": "text",
                                "text": format!("{text}\nans = {summary_str}")
                            }
                        ]
                    }))
                } else {
                    Ok(json!({ "content": [{ "type": "text", "text": "OK" }] }))
                }
            }
            "list_variables" => {
                let out = self.session.execute(EngineCommand::ListVariables)?;
                if let EngineOutput::VariableList(vars) = out {
                    Ok(json!({
                        "content": [
                            {
                                "type": "text",
                                "text": serde_json::to_string_pretty(&vars).map_err(|e| McpError::Json(e.to_string()))?
                            }
                        ]
                    }))
                } else {
                    Ok(json!({ "content": [{ "type": "text", "text": "[]" }] }))
                }
            }
            "get_variable" => {
                let var_name = args.get("name").and_then(Value::as_str).ok_or_else(|| {
                    McpError::InvalidRequest("Missing variable `name`".into())
                })?;
                let out = self.session.execute(EngineCommand::GetVariable {
                    name: var_name.to_string(),
                })?;
                if let EngineOutput::VariableData { name, rows, cols, unit, values } = out {
                    Ok(json!({
                        "content": [
                            {
                                "type": "text",
                                "text": format!("Variable {name}: {rows}x{cols} [{unit}]\nValues: {values:?}")
                            }
                        ]
                    }))
                } else {
                    Ok(json!({ "content": [{ "type": "text", "text": "Variable not found" }] }))
                }
            }
            "render_figure" => {
                let fig_id = args.get("figure_id").and_then(Value::as_u64).ok_or_else(|| {
                    McpError::InvalidRequest("Missing `figure_id`".into())
                })? as usize;
                let w = args.get("width").and_then(Value::as_u64).unwrap_or(800) as u32;
                let h = args.get("height").and_then(Value::as_u64).unwrap_or(600) as u32;
                let out = self.session.execute(EngineCommand::RenderFigureSvg {
                    figure_id: fig_id,
                    width: w,
                    height: h,
                })?;
                if let EngineOutput::Svg(svg_text) = out {
                    Ok(json!({
                        "content": [
                            {
                                "type": "text",
                                "text": svg_text
                            }
                        ]
                    }))
                } else {
                    Ok(json!({ "content": [{ "type": "text", "text": "Figure not found" }] }))
                }
            }
            "run_simulation" => {
                let dur = args.get("duration_sec").and_then(Value::as_f64).ok_or_else(|| {
                    McpError::InvalidRequest("Missing `duration_sec`".into())
                })?;
                let dt = args.get("dt_sec").and_then(Value::as_f64).unwrap_or(0.01);
                let out = self.session.execute(EngineCommand::RunSimulation {
                    duration_sec: dur,
                    dt_sec: dt,
                })?;
                if let EngineOutput::SimResult { times_count, signal_names } = out {
                    Ok(json!({
                        "content": [
                            {
                                "type": "text",
                                "text": format!("Simulation completed: {times_count} timesteps, signals: {signal_names:?}")
                            }
                        ]
                    }))
                } else {
                    Ok(json!({ "content": [{ "type": "text", "text": "Simulation finished" }] }))
                }
            }
            "discover_dynamics" => {
                let xs = args.get("x_samples").and_then(Value::as_array).ok_or_else(|| {
                    McpError::InvalidRequest("Missing `x_samples` array".into())
                })?;
                let dxs = args.get("dx_samples").and_then(Value::as_array).ok_or_else(|| {
                    McpError::InvalidRequest("Missing `dx_samples` array".into())
                })?;
                let thresh = args.get("threshold").and_then(Value::as_f64).unwrap_or(0.001);

                let x_vec: Vec<f64> = xs.iter().filter_map(Value::as_f64).collect();
                let dx_vec: Vec<f64> = dxs.iter().filter_map(Value::as_f64).collect();

                let out = self.session.execute(EngineCommand::DiscoverOdeSindy {
                    x_samples: x_vec,
                    dx_samples: dx_vec,
                    threshold: thresh,
                })?;
                if let EngineOutput::DiscoveredEquation { equation, coefficients } = out {
                    Ok(json!({
                        "content": [
                            {
                                "type": "text",
                                "text": format!("Discovered ODE: {equation}\nCoefficients: {coefficients:?}")
                            }
                        ]
                    }))
                } else {
                    Ok(json!({ "content": [{ "type": "text", "text": "Discovery failed" }] }))
                }
            }
            "panic_hunt" => {
                let iters = args.get("iterations").and_then(Value::as_u64).unwrap_or(100) as usize;
                let out = self.session.execute(EngineCommand::PanicHuntFuzz { iterations: iters })?;
                if let EngineOutput::FuzzAuditPassed { iterations, errors_caught_safely } = out {
                    Ok(json!({
                        "content": [
                            {
                                "type": "text",
                                "text": format!("Fuzz Audit PASSED: {iterations} iterations, {errors_caught_safely} hostile cases handled safely with 0 panics.")
                            }
                        ]
                    }))
                } else {
                    Ok(json!({ "content": [{ "type": "text", "text": "Audit complete" }] }))
                }
            }
            other => Err(McpError::MethodNotFound(format!("Unknown tool `{other}`"))),
        }
    }
}
