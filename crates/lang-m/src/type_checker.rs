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

//! Static type inference for `.m` programs.

use crate::{BinOp, Expr, Stmt};
use std::collections::BTreeMap;

/// Type of a `.m` expression.
#[derive(Clone, Debug, PartialEq)]
pub enum MType {
    Scalar,
    Matrix { rows: usize, cols: usize },
    Unknown,
}

/// Type environment: maps variable names to their inferred `MType`.
pub type TypeEnv = BTreeMap<String, MType>;

/// Type-level warnings generated during static analysis.
#[derive(Clone, Debug, PartialEq)]
pub enum TypeWarning {
    /// Shape mismatch in an operation.
    ShapeMismatch { op: String, lhs: MType, rhs: MType },
    /// Type of a variable is unknown (not yet inferred).
    UnknownType(String),
}

/// Infer the type of a single expression.
pub fn infer_expr(expr: &Expr, env: &TypeEnv) -> MType {
    match expr {
        Expr::Literal(_, _) => MType::Scalar,
        Expr::Matrix(rows) => {
            let nrows = rows.len();
            if nrows == 0 {
                return MType::Matrix { rows: 0, cols: 0 };
            }
            let ncols = rows.first().map(|r| r.len()).unwrap_or(0);
            MType::Matrix { rows: nrows, cols: ncols }
        }
        Expr::Var(name) => env
            .get(name)
            .cloned()
            .unwrap_or(MType::Unknown),
        Expr::Binary { op, lhs, rhs } => infer_binary(op, lhs, rhs, env),
        Expr::Transpose(inner) => match infer_expr(inner, env) {
            MType::Matrix { rows, cols } => MType::Matrix { rows: cols, cols: rows },
            MType::Scalar => MType::Scalar,
            MType::Unknown => MType::Unknown,
        },
        Expr::UnaryNeg(inner) | Expr::LogicalNot(inner) => infer_expr(inner, env),
        Expr::CallOrIndex { .. } => MType::Unknown,
        Expr::AnonFunc { .. } => MType::Unknown,
        Expr::Range { .. } => MType::Unknown,
        Expr::EndKeyword | Expr::ColonAll => MType::Unknown,
    }
}

/// Infer the result type of a binary operation.
fn infer_binary(op: &BinOp, lhs: &Expr, rhs: &Expr, env: &TypeEnv) -> MType {
    let lt = infer_expr(lhs, env);
    let rt = infer_expr(rhs, env);
    match op {
        BinOp::MatMul => {
            match (&lt, &rt) {
                (MType::Matrix { rows: lr, cols: lc }, MType::Matrix { rows: rr, cols: rc }) => {
                    if lc == rr {
                        MType::Matrix { rows: *lr, cols: *rc }
                    } else {
                        MType::Unknown
                    }
                }
                (MType::Scalar, t) | (t, MType::Scalar) => t.clone(),
                _ => MType::Unknown,
            }
        }
        BinOp::Add | BinOp::Sub => {
            match (&lt, &rt) {
                (MType::Scalar, MType::Scalar) => MType::Scalar,
                (MType::Matrix { rows: lr, cols: lc }, MType::Matrix { rows: rr, cols: rc })
                    if lr == rr && lc == rc =>
                {
                    lt.clone()
                }
                (MType::Scalar, m) | (m, MType::Scalar) => m.clone(),
                _ => MType::Unknown,
            }
        }
        BinOp::ElemMul | BinOp::ElemDiv | BinOp::ElemPow => {
            match (&lt, &rt) {
                (MType::Scalar, MType::Scalar) => MType::Scalar,
                (MType::Matrix { .. }, MType::Matrix { .. }) => lt.clone(),
                (MType::Scalar, m) | (m, MType::Scalar) => m.clone(),
                _ => MType::Unknown,
            }
        }
        _ => MType::Unknown,
    }
}

/// Type-check a sequence of statements, updating `env` with assignments.
/// Returns a list of type warnings.
pub fn type_check_program(stmts: &[Stmt], env: &mut TypeEnv) -> Vec<TypeWarning> {
    let mut warnings = Vec::new();
    for stmt in stmts {
        match stmt {
            Stmt::Assign { target, expr, .. } => {
                let ty = infer_expr(expr, env);
                if ty == MType::Unknown {
                    warnings.push(TypeWarning::UnknownType(target.clone()));
                }
                env.insert(target.clone(), ty);
            }
            Stmt::MultiAssign { targets, expr, .. } => {
                let ty = infer_expr(expr, env);
                for t in targets {
                    if ty == MType::Unknown {
                        warnings.push(TypeWarning::UnknownType(t.clone()));
                    }
                    env.insert(t.clone(), ty.clone());
                }
            }
            Stmt::Expr { expr, .. } => {
                let ty = infer_expr(expr, env);
                // Check binary operations for shape mismatches
                if let Expr::Binary { op, lhs, rhs } = expr {
                    let lt = infer_expr(lhs, env);
                    let rt = infer_expr(rhs, env);
                    check_binary_shapes(op, &lt, &rt, &ty, &mut warnings);
                }
                let _ = ty;
            }
            Stmt::If { branches, else_body } => {
                for (_, body) in branches {
                    warnings.extend(type_check_program(body, env));
                }
                warnings.extend(type_check_program(else_body, env));
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } => {
                warnings.extend(type_check_program(body, env));
            }
            Stmt::Switch { cases, otherwise, .. } => {
                for (_, body) in cases {
                    warnings.extend(type_check_program(body, env));
                }
                warnings.extend(type_check_program(otherwise, env));
            }
            Stmt::FunctionDef(f) => {
                let mut fn_env = env.clone();
                for p in &f.params {
                    fn_env.insert(p.clone(), MType::Unknown);
                }
                warnings.extend(type_check_program(&f.body, &mut fn_env));
            }
            Stmt::IndexedAssign { target, expr, .. } => {
                let ty = infer_expr(expr, env);
                env.insert(target.clone(), ty);
            }
            Stmt::Return | Stmt::Break | Stmt::Continue => {}
        }
    }
    warnings
}

fn check_binary_shapes(
    op: &BinOp,
    lt: &MType,
    rt: &MType,
    result: &MType,
    warnings: &mut Vec<TypeWarning>,
) {
    if *result == MType::Unknown {
        match (lt, rt) {
            (MType::Matrix { cols: lc, .. }, MType::Matrix { rows: rr, .. })
                if matches!(op, BinOp::MatMul) && lc != rr =>
            {
                warnings.push(TypeWarning::ShapeMismatch {
                    op: format!("{op:?}"),
                    lhs: lt.clone(),
                    rhs: rt.clone(),
                });
            }
            (MType::Matrix { rows: lr, cols: lc }, MType::Matrix { rows: rr, cols: rc })
                if matches!(op, BinOp::Add | BinOp::Sub) && (lr != rr || lc != rc) =>
            {
                warnings.push(TypeWarning::ShapeMismatch {
                    op: format!("{op:?}"),
                    lhs: lt.clone(),
                    rhs: rt.clone(),
                });
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BinOp, Expr, SiUnit, Stmt};

    #[test]
    fn test_infer_literal_and_var() {
        let env = TypeEnv::new();
        let lit = Expr::Literal(42.0, SiUnit::DIMENSIONLESS);
        assert_eq!(infer_expr(&lit, &env), MType::Scalar);

        let var = Expr::Var("x".to_string());
        assert_eq!(infer_expr(&var, &env), MType::Unknown);

        let mut env2 = TypeEnv::new();
        env2.insert("x".to_string(), MType::Scalar);
        assert_eq!(infer_expr(&var, &env2), MType::Scalar);
    }

    #[test]
    fn test_infer_matrix_literal() {
        let env = TypeEnv::new();
        let mat = Expr::Matrix(vec![
            vec![Expr::Literal(1.0, SiUnit::DIMENSIONLESS), Expr::Literal(2.0, SiUnit::DIMENSIONLESS)],
            vec![Expr::Literal(3.0, SiUnit::DIMENSIONLESS), Expr::Literal(4.0, SiUnit::DIMENSIONLESS)],
        ]);
        assert_eq!(infer_expr(&mat, &env), MType::Matrix { rows: 2, cols: 2 });
    }

    #[test]
    fn test_matmul_shape_propagation() {
        let mut env = TypeEnv::new();
        env.insert("A".to_string(), MType::Matrix { rows: 2, cols: 3 });
        env.insert("B".to_string(), MType::Matrix { rows: 3, cols: 4 });
        let expr = Expr::Binary {
            op: BinOp::MatMul,
            lhs: Box::new(Expr::Var("A".to_string())),
            rhs: Box::new(Expr::Var("B".to_string())),
        };
        assert_eq!(infer_expr(&expr, &env), MType::Matrix { rows: 2, cols: 4 });
    }

    #[test]
    fn test_type_check_program_assign() {
        let mut env = TypeEnv::new();
        let stmts = vec![
            Stmt::Assign {
                target: "x".to_string(),
                expr: Expr::Literal(1.0, SiUnit::DIMENSIONLESS),
                suppress_output: false,
            },
        ];
        let warns = type_check_program(&stmts, &mut env);
        assert!(warns.is_empty());
        assert_eq!(env.get("x"), Some(&MType::Scalar));
    }

    #[test]
    fn test_type_check_unknown_var_warns() {
        let mut env = TypeEnv::new();
        let stmts = vec![
            Stmt::Assign {
                target: "y".to_string(),
                expr: Expr::Var("unknown_var".to_string()),
                suppress_output: false,
            },
        ];
        let warns = type_check_program(&stmts, &mut env);
        assert!(!warns.is_empty());
        assert!(matches!(warns.first(), Some(TypeWarning::UnknownType(_))));
    }
}
