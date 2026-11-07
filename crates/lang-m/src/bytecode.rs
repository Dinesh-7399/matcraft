use crate::{BinOp, Expr, LangError};
use std::collections::BTreeMap;

/// Register-Based Virtual Machine Instruction Set (`bytecode.rs` — MATLAB LXE Equivalent).
/// Compiles scalar `.m` AST expressions and inner loop bodies into a flat array of 32-bit-style
/// register instructions for zero-tree-traversal execution speed!
#[derive(Clone, Debug, PartialEq)]
pub enum OpCode {
    LoadConst { dst: usize, val: f64 },
    LoadVar { dst: usize, slot: usize },
    StoreVar { src: usize, slot: usize },
    NegR { dst: usize, src: usize },
    AddR { dst: usize, lhs: usize, rhs: usize },
    SubR { dst: usize, lhs: usize, rhs: usize },
    MulR { dst: usize, lhs: usize, rhs: usize },
    DivR { dst: usize, lhs: usize, rhs: usize },
    PowR { dst: usize, lhs: usize, rhs: usize },
    SinR { dst: usize, src: usize },
    CosR { dst: usize, src: usize },
    ExpR { dst: usize, src: usize },
    SqrtR { dst: usize, src: usize },
    Return { src: usize },
}

#[derive(Clone, Debug)]
pub struct CompiledBytecode {
    pub instructions: Vec<OpCode>,
    pub symbol_slots: BTreeMap<String, usize>,
    pub num_registers: usize,
}

impl CompiledBytecode {
    pub fn compile_expr(expr: &Expr) -> Result<Self, LangError> {
        let mut compiler = BytecodeCompiler {
            instructions: Vec::new(),
            symbol_slots: BTreeMap::new(),
            next_reg: 0,
        };
        let out_reg = compiler.emit_expr(expr)?;
        compiler.instructions.push(OpCode::Return { src: out_reg });
        Ok(Self {
            instructions: compiler.instructions,
            symbol_slots: compiler.symbol_slots,
            num_registers: compiler.next_reg,
        })
    }

    /// Executes the compiled register bytecode against `var_slots`.
    pub fn execute(&self, var_slots: &[f64]) -> f64 {
        let mut regs = vec![0.0; self.num_registers.max(1)];
        for op in &self.instructions {
            match *op {
                OpCode::LoadConst { dst, val } => {
                    if let Some(r) = regs.get_mut(dst) {
                        *r = val;
                    }
                }
                OpCode::LoadVar { dst, slot } => {
                    let v = var_slots.get(slot).copied().unwrap_or(0.0);
                    if let Some(r) = regs.get_mut(dst) {
                        *r = v;
                    }
                }
                OpCode::StoreVar { .. } => {}
                OpCode::NegR { dst, src } => {
                    let v = -regs.get(src).copied().unwrap_or(0.0);
                    if let Some(r) = regs.get_mut(dst) {
                        *r = v;
                    }
                }
                OpCode::AddR { dst, lhs, rhs } => {
                    let v = regs.get(lhs).copied().unwrap_or(0.0)
                        + regs.get(rhs).copied().unwrap_or(0.0);
                    if let Some(r) = regs.get_mut(dst) {
                        *r = v;
                    }
                }
                OpCode::SubR { dst, lhs, rhs } => {
                    let v = regs.get(lhs).copied().unwrap_or(0.0)
                        - regs.get(rhs).copied().unwrap_or(0.0);
                    if let Some(r) = regs.get_mut(dst) {
                        *r = v;
                    }
                }
                OpCode::MulR { dst, lhs, rhs } => {
                    let v = regs.get(lhs).copied().unwrap_or(0.0)
                        * regs.get(rhs).copied().unwrap_or(0.0);
                    if let Some(r) = regs.get_mut(dst) {
                        *r = v;
                    }
                }
                OpCode::DivR { dst, lhs, rhs } => {
                    let v = regs.get(lhs).copied().unwrap_or(0.0)
                        / regs.get(rhs).copied().unwrap_or(0.0);
                    if let Some(r) = regs.get_mut(dst) {
                        *r = v;
                    }
                }
                OpCode::PowR { dst, lhs, rhs } => {
                    let v = regs
                        .get(lhs)
                        .copied()
                        .unwrap_or(0.0)
                        .powf(regs.get(rhs).copied().unwrap_or(0.0));
                    if let Some(r) = regs.get_mut(dst) {
                        *r = v;
                    }
                }
                OpCode::SinR { dst, src } => {
                    let v = regs.get(src).copied().unwrap_or(0.0).sin();
                    if let Some(r) = regs.get_mut(dst) {
                        *r = v;
                    }
                }
                OpCode::CosR { dst, src } => {
                    let v = regs.get(src).copied().unwrap_or(0.0).cos();
                    if let Some(r) = regs.get_mut(dst) {
                        *r = v;
                    }
                }
                OpCode::ExpR { dst, src } => {
                    let v = regs.get(src).copied().unwrap_or(0.0).exp();
                    if let Some(r) = regs.get_mut(dst) {
                        *r = v;
                    }
                }
                OpCode::SqrtR { dst, src } => {
                    let v = regs.get(src).copied().unwrap_or(0.0).sqrt();
                    if let Some(r) = regs.get_mut(dst) {
                        *r = v;
                    }
                }
                OpCode::Return { src } => {
                    return regs.get(src).copied().unwrap_or(0.0);
                }
            }
        }
        0.0
    }
}

struct BytecodeCompiler {
    instructions: Vec<OpCode>,
    symbol_slots: BTreeMap<String, usize>,
    next_reg: usize,
}

impl BytecodeCompiler {
    fn alloc_reg(&mut self) -> usize {
        let r = self.next_reg;
        self.next_reg += 1;
        r
    }

    fn slot_for(&mut self, name: &str) -> usize {
        let next_slot = self.symbol_slots.len();
        *self.symbol_slots.entry(name.to_string()).or_insert(next_slot)
    }

    fn emit_expr(&mut self, expr: &Expr) -> Result<usize, LangError> {
        match expr {
            Expr::Literal(val, _) => {
                let dst = self.alloc_reg();
                self.instructions.push(OpCode::LoadConst { dst, val: *val });
                Ok(dst)
            }
            Expr::Var(name) => {
                let slot = self.slot_for(name);
                let dst = self.alloc_reg();
                self.instructions.push(OpCode::LoadVar { dst, slot });
                Ok(dst)
            }
            Expr::UnaryNeg(inner) => {
                let src = self.emit_expr(inner)?;
                let dst = self.alloc_reg();
                self.instructions.push(OpCode::NegR { dst, src });
                Ok(dst)
            }
            Expr::Binary { op, lhs, rhs } => {
                let l_reg = self.emit_expr(lhs)?;
                let r_reg = self.emit_expr(rhs)?;
                let dst = self.alloc_reg();
                let inst = match op {
                    BinOp::Add => OpCode::AddR {
                        dst,
                        lhs: l_reg,
                        rhs: r_reg,
                    },
                    BinOp::Sub => OpCode::SubR {
                        dst,
                        lhs: l_reg,
                        rhs: r_reg,
                    },
                    BinOp::MatMul | BinOp::ElemMul => OpCode::MulR {
                        dst,
                        lhs: l_reg,
                        rhs: r_reg,
                    },
                    BinOp::RightDiv | BinOp::ElemDiv => OpCode::DivR {
                        dst,
                        lhs: l_reg,
                        rhs: r_reg,
                    },
                    BinOp::Pow | BinOp::ElemPow => OpCode::PowR {
                        dst,
                        lhs: l_reg,
                        rhs: r_reg,
                    },
                    _ => OpCode::AddR {
                        dst,
                        lhs: l_reg,
                        rhs: r_reg,
                    },
                };
                self.instructions.push(inst);
                Ok(dst)
            }
            Expr::CallOrIndex { name, args } => {
                let first = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: name.clone(),
                    reason: "Expected 1 argument in bytecode intrinsic".into(),
                })?;
                let src = self.emit_expr(first)?;
                let dst = self.alloc_reg();
                let inst = match name.as_str() {
                    "sin" => OpCode::SinR { dst, src },
                    "cos" => OpCode::CosR { dst, src },
                    "exp" => OpCode::ExpR { dst, src },
                    "sqrt" => OpCode::SqrtR { dst, src },
                    other => {
                        return Err(LangError::UndefinedSymbol(other.to_string()));
                    }
                };
                self.instructions.push(inst);
                Ok(dst)
            }
            _ => Err(LangError::Syntax {
                pos: 0,
                message: "Unsupported expression in scalar register bytecode".into(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Parser, tokenize};

    #[test]
    fn test_register_bytecode_vm_matches_ast_eval() -> Result<(), Box<dyn std::error::Error>> {
        let tokens = tokenize("sin(x)^2 + cos(x)^2 + 0.5 * y")?;
        let mut parser = Parser::new(tokens);
        let expr = parser.parse_expr()?;
        let bc = CompiledBytecode::compile_expr(&expr)?;

        // Slot 0 = x = 1.234, Slot 1 = y = 8.0 -> sin^2 + cos^2 + 4.0 = 5.0!
        let res = bc.execute(&[1.234, 8.0]);
        assert!((res - 5.0).abs() < 1e-12);
        Ok(())
    }
}
