use crate::{BinOp, Expr, Parser, Stmt, tokenize};
use num_core::SiUnit;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticDiagnostic {
    pub variable: String,
    pub message: String,
    pub inferred_unit: Option<String>,
}

/// Static Hindley-Milner style SI-Unit & Dimensional Inference Pass (`hir.rs`).
/// Analyzes an `.m` script *before* execution to surface live unit inlay hints and dimensional errors!
pub fn infer_program_units(
    source: &str,
) -> (BTreeMap<String, SiUnit>, Vec<StaticDiagnostic>) {
    let mut env: BTreeMap<String, SiUnit> = BTreeMap::new();
    let mut diags = Vec::new();

    let Ok(tokens) = tokenize(source) else {
        return (env, diags);
    };
    let mut parser = Parser::new(tokens);
    let Ok(stmts) = parser.parse_program() else {
        return (env, diags);
    };

    for stmt in &stmts {
        infer_stmt_units(stmt, &mut env, &mut diags);
    }
    (env, diags)
}

fn infer_stmt_units(
    stmt: &Stmt,
    env: &mut BTreeMap<String, SiUnit>,
    diags: &mut Vec<StaticDiagnostic>,
) {
    match stmt {
        Stmt::Assign { target, expr, .. } => {
            match infer_expr_unit(expr, env) {
                Ok(u) => {
                    env.insert(target.clone(), u);
                }
                Err(msg) => {
                    diags.push(StaticDiagnostic {
                        variable: target.clone(),
                        message: msg,
                        inferred_unit: None,
                    });
                }
            }
        }
        Stmt::If {
            branches,
            else_body,
        } => {
            for (_, body) in branches {
                for s in body {
                    infer_stmt_units(s, env, diags);
                }
            }
            for s in else_body {
                infer_stmt_units(s, env, diags);
            }
        }
        Stmt::For { var, range, body } => {
            if let Ok(u) = infer_expr_unit(range, env) {
                env.insert(var.clone(), u);
            }
            for s in body {
                infer_stmt_units(s, env, diags);
            }
        }
        _ => {}
    }
}

fn infer_expr_unit(expr: &Expr, env: &BTreeMap<String, SiUnit>) -> Result<SiUnit, String> {
    match expr {
        Expr::Literal(_, u) => Ok(*u),
        Expr::Var(name) => env
            .get(name)
            .copied()
            .ok_or_else(|| format!("Unknown variable `{name}`")),
        Expr::UnaryNeg(inner) | Expr::Transpose(inner) => infer_expr_unit(inner, env),
        Expr::Binary { op, lhs, rhs } => {
            let lu = infer_expr_unit(lhs, env)?;
            let ru = infer_expr_unit(rhs, env)?;
            match op {
                BinOp::Add | BinOp::Sub => lu
                    .verify_same(ru, "+/-")
                    .map_err(|e| e.to_string()),
                BinOp::MatMul | BinOp::ElemMul => lu
                    .checked_mul(ru)
                    .map_err(|e| e.to_string()),
                BinOp::RightDiv | BinOp::ElemDiv => lu
                    .checked_div(ru)
                    .map_err(|e| e.to_string()),
                BinOp::LeftDiv => ru
                    .checked_div(lu)
                    .map_err(|e| e.to_string()),
                _ => Ok(SiUnit::DIMENSIONLESS),
            }
        }
        _ => Ok(SiUnit::DIMENSIONLESS),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Interpreter, LangError, NoopExternalProvider};
    use num_core::MemoryBudget;
    use std::sync::Arc;

    #[test]
    fn test_user_functions_multi_return_end_slicing_and_hir_inference(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        let mut interp = Interpreter::new(Arc::clone(&budget));
        let mut ext = NoopExternalProvider;

        // 1. Multi-output user function + anonymous handle + `end` indexing + colon slicing
        let script = r#"
            function [kin, mom] = mechanics(m, v)
                kin = 0.5 * m * v^2;
                mom = m * v;
            end
            [E, p] = mechanics(4_kg, 3_m/s);
            sq = @(x) x * x;
            arr = [10, 20, 30, 40, 50];
            sub = arr(1, 2:end-1);
            s_val = sum(sub) + sq(5);
        "#;
        interp.eval_script(script, &mut ext)?;
        let e_val = interp.vars.get("E").ok_or("Missing E")?;
        let p_val = interp.vars.get("p").ok_or("Missing p")?;
        let s_val = interp.vars.get("s_val").ok_or("Missing s_val")?;

        assert_eq!(e_val.get(0, 0)?, 18.0);
        assert_eq!(e_val.unit(), SiUnit::JOULE);
        assert_eq!(p_val.get(0, 0)?, 12.0);
        assert_eq!(s_val.get(0, 0)?, (20.0 + 30.0 + 40.0) + 25.0);

        // 2. Static HIR Unit Inference catches dimensional bug without executing!
        let (units, diags) = infer_program_units("d = 100_m; t = 5_s; v = d / t; bad = d + t;");
        assert_eq!(units.get("v"), Some(&SiUnit::parse("m/s")?));
        assert_eq!(diags.len(), 1);

        // 3. Test recursion limit protection prevents thread stack overflow crash
        let rec_script = r#"
            function y = rec_crash(x)
                y = rec_crash(x);
            end
            res = rec_crash(1);
        "#;
        let mut rec_interp = Interpreter::new(Arc::clone(&budget));
        let rec_res = rec_interp.eval_script(rec_script, &mut ext);
        assert!(matches!(rec_res, Err(LangError::RecursionLimitExceeded { .. })));

        // 4. Test unit lexing with subtraction and arithmetic: 10_m - 2_m
        let unit_arith = "diff = 10_m - 2_m; prod = 5_m * 3_s;";
        let mut u_interp = Interpreter::new(Arc::clone(&budget));
        u_interp.eval_script(unit_arith, &mut ext)?;
        assert_eq!(u_interp.vars.get("diff").map(|t| t.get(0, 0).unwrap_or(0.0)), Some(8.0));
        assert_eq!(u_interp.vars.get("prod").map(|t| t.get(0, 0).unwrap_or(0.0)), Some(15.0));

        // 5. Test multi-element indexed assignment: A(1:2, 1:2) = [10, 20; 30, 40]
        let slice_assign = r#"
            A = [1, 2; 3, 4];
            A(1:2, 1:2) = [10, 20; 30, 40];
        "#;
        let mut sa_interp = Interpreter::new(Arc::clone(&budget));
        sa_interp.eval_script(slice_assign, &mut ext)?;
        let a_mat = sa_interp.vars.get("A").ok_or("Missing A")?;
        assert_eq!(a_mat.get(0, 0)?, 10.0);
        assert_eq!(a_mat.get(0, 1)?, 20.0);
        assert_eq!(a_mat.get(1, 0)?, 30.0);
        assert_eq!(a_mat.get(1, 1)?, 40.0);

        // 6. Test matrix power (^) vs element-wise power (.^)
        let pow_script = r#"
            M = [1, 2; 3, 4];
            M_pow2 = M^2;
            M_epow2 = M.^2;
        "#;
        let mut pow_interp = Interpreter::new(Arc::clone(&budget));
        pow_interp.eval_script(pow_script, &mut ext)?;
        let m_pow2 = pow_interp.vars.get("M_pow2").ok_or("Missing M_pow2")?;
        let m_epow2 = pow_interp.vars.get("M_epow2").ok_or("Missing M_epow2")?;
        // M^2 = [1, 2; 3, 4] * [1, 2; 3, 4] = [7, 10; 15, 22]
        assert_eq!(m_pow2.get(0, 0)?, 7.0);
        assert_eq!(m_pow2.get(0, 1)?, 10.0);
        assert_eq!(m_pow2.get(1, 0)?, 15.0);
        assert_eq!(m_pow2.get(1, 1)?, 22.0);
        // M.^2 = [1, 4; 9, 16]
        assert_eq!(m_epow2.get(0, 0)?, 1.0);
        assert_eq!(m_epow2.get(0, 1)?, 4.0);
        assert_eq!(m_epow2.get(1, 0)?, 9.0);
        assert_eq!(m_epow2.get(1, 1)?, 16.0);

        Ok(())
    }
}
