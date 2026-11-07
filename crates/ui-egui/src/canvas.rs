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

//! Graphical Simulink Block Diagram Canvas View-Model.
//! Supports node-graph block placement, input/output port wiring,
//! topological acyclicity checks, and direct compilation to a runnable `SimModel`.

use crate::UiError;
use serde::{Deserialize, Serialize};
use sim_engine::{BlockKind, SimBlock, SimModel};
use std::collections::BTreeMap;

/// Kinds of blocks supported in the visual canvas.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CanvasBlockKind {
    Constant { value: f64 },
    Integrator { initial: f64 },
    Gain { k: f64 },
    Sum { signs: Vec<f64> },
    TransferFcn { b0: f64, b1: f64, a1: f64, a0: f64 },
    Scope,
}

/// An input or output port on a canvas block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanvasPort {
    pub id: usize,
    pub name: String,
    pub is_output: bool,
    pub relative_pos: [f32; 2],
}

/// A visual block element in the node diagram.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanvasBlock {
    pub id: usize,
    pub name: String,
    pub kind: CanvasBlockKind,
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub inputs: Vec<CanvasPort>,
    pub outputs: Vec<CanvasPort>,
}

/// A directed wire connecting a source output port to a destination input port.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanvasWire {
    pub id: usize,
    pub src_block: usize,
    pub src_port: usize,
    pub dst_block: usize,
    pub dst_port: usize,
}

/// The interactive block canvas state.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BlockCanvas {
    pub blocks: BTreeMap<usize, CanvasBlock>,
    pub wires: Vec<CanvasWire>,
    pub next_id: usize,
    pub selected_block: Option<usize>,
    pub pan_offset: [f32; 2],
    pub zoom: f32,
}

impl BlockCanvas {
    /// Creates a new empty canvas.
    #[must_use]
    pub fn new() -> Self {
        Self {
            blocks: BTreeMap::new(),
            wires: Vec::new(),
            next_id: 1,
            selected_block: None,
            pan_offset: [0.0, 0.0],
            zoom: 1.0,
        }
    }

    /// Allocates a new unique identifier.
    fn alloc_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    /// Adds a block to the diagram at the specified position.
    pub fn add_block(
        &mut self,
        name: impl Into<String>,
        kind: CanvasBlockKind,
        position: [f32; 2],
    ) -> usize {
        let id = self.alloc_id();
        let (inputs, outputs) = match &kind {
            CanvasBlockKind::Constant { .. } => (
                Vec::new(),
                vec![CanvasPort {
                    id: 0,
                    name: "out".into(),
                    is_output: true,
                    relative_pos: [100.0, 25.0],
                }],
            ),
            CanvasBlockKind::Integrator { .. }
            | CanvasBlockKind::Gain { .. }
            | CanvasBlockKind::TransferFcn { .. } => (
                vec![CanvasPort {
                    id: 0,
                    name: "in".into(),
                    is_output: false,
                    relative_pos: [0.0, 25.0],
                }],
                vec![CanvasPort {
                    id: 0,
                    name: "out".into(),
                    is_output: true,
                    relative_pos: [100.0, 25.0],
                }],
            ),
            CanvasBlockKind::Sum { signs } => {
                let mut ins = Vec::with_capacity(signs.len());
                let step = 50.0 / (signs.len() as f32).max(1.0);
                for (i, &s) in signs.iter().enumerate() {
                    let label = if s >= 0.0 { "+" } else { "-" };
                    ins.push(CanvasPort {
                        id: i,
                        name: label.to_string(),
                        is_output: false,
                        relative_pos: [0.0, (i as f32) * step + 15.0],
                    });
                }
                (
                    ins,
                    vec![CanvasPort {
                        id: 0,
                        name: "out".into(),
                        is_output: true,
                        relative_pos: [60.0, 25.0],
                    }],
                )
            }
            CanvasBlockKind::Scope => (
                vec![CanvasPort {
                    id: 0,
                    name: "in".into(),
                    is_output: false,
                    relative_pos: [0.0, 25.0],
                }],
                Vec::new(),
            ),
        };

        let block = CanvasBlock {
            id,
            name: name.into(),
            kind,
            position,
            size: [100.0, 50.0],
            inputs,
            outputs,
        };

        self.blocks.insert(id, block);
        id
    }

    /// Connects two blocks with a directed signal wire.
    pub fn connect(
        &mut self,
        src_block: usize,
        src_port: usize,
        dst_block: usize,
        dst_port: usize,
    ) -> Result<usize, UiError> {
        if !self.blocks.contains_key(&src_block) {
            return Err(UiError::State(format!("Source block {src_block} not found")));
        }
        if !self.blocks.contains_key(&dst_block) {
            return Err(UiError::State(format!(
                "Destination block {dst_block} not found"
            )));
        }

        let wire_id = self.alloc_id();
        self.wires.push(CanvasWire {
            id: wire_id,
            src_block,
            src_port,
            dst_block,
            dst_port,
        });

        Ok(wire_id)
    }

    /// Removes a wire by ID.
    pub fn disconnect(&mut self, wire_id: usize) -> bool {
        let initial_len = self.wires.len();
        self.wires.retain(|w| w.id != wire_id);
        self.wires.len() < initial_len
    }

    /// Moves a block to a new position.
    pub fn move_block(&mut self, block_id: usize, new_pos: [f32; 2]) -> bool {
        if let Some(blk) = self.blocks.get_mut(&block_id) {
            blk.position = new_pos;
            true
        } else {
            false
        }
    }

    /// Computes topological execution order of blocks. Returns error if an unbroken algebraic cycle exists.
    pub fn compute_execution_order(&self) -> Result<Vec<usize>, UiError> {
        let mut in_degree: BTreeMap<usize, usize> = BTreeMap::new();
        let mut adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

        for &id in self.blocks.keys() {
            in_degree.insert(id, 0);
            adj.insert(id, Vec::new());
        }

        for w in &self.wires {
            // State integrators break instantaneous algebraic dependency
            let is_dynamic_delay = self
                .blocks
                .get(&w.src_block)
                .is_some_and(|b| matches!(b.kind, CanvasBlockKind::Integrator { .. }));

            if !is_dynamic_delay {
                if let Some(entry) = in_degree.get_mut(&w.dst_block) {
                    *entry = entry.saturating_add(1);
                }
                if let Some(list) = adj.get_mut(&w.src_block) {
                    list.push(w.dst_block);
                }
            }
        }

        let mut queue: Vec<usize> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut order = Vec::with_capacity(self.blocks.len());

        while let Some(u) = queue.pop() {
            order.push(u);
            if let Some(neighbors) = adj.get(&u) {
                for &v in neighbors {
                    if let Some(deg) = in_degree.get_mut(&v) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push(v);
                        }
                    }
                }
            }
        }

        if order.len() < self.blocks.len() {
            return Err(UiError::State(
                "Direct feedthrough algebraic loop detected in diagram".into(),
            ));
        }

        Ok(order)
    }

    /// Transpiles the graphical canvas model into a continuous/discrete `SimModel`.
    pub fn to_sim_model(&self) -> Result<SimModel, UiError> {
        let mut model = SimModel::new();

        for (&id, blk) in &self.blocks {
            let sim_kind = match &blk.kind {
                CanvasBlockKind::Constant { value } => BlockKind::Constant { value: *value },
                CanvasBlockKind::Integrator { initial } => BlockKind::Integrator {
                    initial_state: *initial,
                },
                CanvasBlockKind::Gain { k } => BlockKind::Gain { gain: *k },
                CanvasBlockKind::Sum { signs } => BlockKind::Sum {
                    signs: signs.clone(),
                },
                CanvasBlockKind::TransferFcn { b0, b1, a1, a0 } => BlockKind::TransferFcn {
                    b0: *b0,
                    b1: *b1,
                    a1: *a1,
                    a0: *a0,
                },
                CanvasBlockKind::Scope => BlockKind::Scope,
            };

            let sim_blk = SimBlock {
                id,
                name: blk.name.clone(),
                kind: sim_kind,
                out_unit: num_core::SiUnit::DIMENSIONLESS,
            };

            model.add_block(sim_blk);
        }

        for w in &self.wires {
            model
                .connect(w.src_block, w.dst_block, w.dst_port)
                .map_err(|e| UiError::State(format!("Wire connection error: {e}")))?;
        }

        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canvas_build_feedback_loop_and_transpile() -> Result<(), UiError> {
        let mut canvas = BlockCanvas::new();

        // 1. Create a closed-loop first-order step response system:
        // Setpoint (Constant: 1.0) -> Error (Sum: + -) -> Integrator (1/s) -> Output (Scope)
        // Feedback from Integrator to Error negative terminal
        let b_const = canvas.add_block("Step", CanvasBlockKind::Constant { value: 1.0 }, [0.0, 0.0]);
        let b_sum = canvas.add_block(
            "Sum",
            CanvasBlockKind::Sum {
                signs: vec![1.0, -1.0],
            },
            [150.0, 0.0],
        );
        let b_integ = canvas.add_block(
            "Integrator",
            CanvasBlockKind::Integrator { initial: 0.0 },
            [300.0, 0.0],
        );
        let b_scope = canvas.add_block(
            "Scope",
            CanvasBlockKind::Scope,
            [450.0, 0.0],
        );

        // Forward connections
        canvas.connect(b_const, 0, b_sum, 0)?;
        canvas.connect(b_sum, 0, b_integ, 0)?;
        canvas.connect(b_integ, 0, b_scope, 0)?;

        // Feedback connection (Integrator output -> Sum negative input 1)
        canvas.connect(b_integ, 0, b_sum, 1)?;

        // Check topological sort (integrator breaks algebraic loop)
        let order = canvas.compute_execution_order()?;
        assert_eq!(order.len(), 4);

        // Convert to runnable SimModel
        let model = canvas.to_sim_model()?;
        assert_eq!(model.blocks.len(), 4);
        assert_eq!(model.wires.len(), 4);

        Ok(())
    }
}
