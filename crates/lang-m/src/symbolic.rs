use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// Symbolic Mathematical Expression Tree (`SymExpr` — Symbolic Math Toolbox equivalent).
/// Supports algebraic simplification (`simplify`), exact analytical symbolic differentiation
/// (`diff(f, var)`), symbolic Taylor series expansion (`taylor`), and numeric evaluation (`subs`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SymExpr {
    Const(f64),
    Var(String),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Pow(Box<Self>, Box<Self>),
    Sin(Box<Self>),
    Cos(Box<Self>),
    Exp(Box<Self>),
    Ln(Box<Self>),
}

impl SymExpr {
    #[must_use]
    pub fn var(name: &str) -> Self {
        Self::Var(name.to_string())
    }

    #[must_use]
    pub const fn constant(c: f64) -> Self {
        Self::Const(c)
    }

    /// Evaluates the symbolic expression numerically given variable bindings (`subs`).
    #[must_use]
    pub fn eval(&self, env: &BTreeMap<String, f64>) -> f64 {
        match self {
            Self::Const(c) => *c,
            Self::Var(v) => env.get(v).copied().unwrap_or(f64::NAN),
            Self::Add(a, b) => a.eval(env) + b.eval(env),
            Self::Sub(a, b) => a.eval(env) - b.eval(env),
            Self::Mul(a, b) => a.eval(env) * b.eval(env),
            Self::Div(a, b) => a.eval(env) / b.eval(env),
            Self::Pow(a, b) => a.eval(env).powf(b.eval(env)),
            Self::Sin(a) => a.eval(env).sin(),
            Self::Cos(a) => a.eval(env).cos(),
            Self::Exp(a) => a.eval(env).exp(),
            Self::Ln(a) => a.eval(env).ln(),
        }
    }

    /// Exact Symbolic Differentiation $\frac{\partial}{\partial \text{var}} f$ (`diff(expr, var)`).
    #[must_use]
    pub fn diff(&self, var: &str) -> Self {
        let raw = match self {
            Self::Const(_) => Self::Const(0.0),
            Self::Var(v) => {
                if v == var {
                    Self::Const(1.0)
                } else {
                    Self::Const(0.0)
                }
            }
            Self::Add(u, v) => Self::Add(Box::new(u.diff(var)), Box::new(v.diff(var))),
            Self::Sub(u, v) => Self::Sub(Box::new(u.diff(var)), Box::new(v.diff(var))),
            // Product rule: (u * v)' = u' * v + u * v'
            Self::Mul(u, v) => Self::Add(
                Box::new(Self::Mul(Box::new(u.diff(var)), v.clone())),
                Box::new(Self::Mul(u.clone(), Box::new(v.diff(var)))),
            ),
            // Quotient rule: (u / v)' = (u' * v - u * v') / v^2
            Self::Div(u, v) => Self::Div(
                Box::new(Self::Sub(
                    Box::new(Self::Mul(Box::new(u.diff(var)), v.clone())),
                    Box::new(Self::Mul(u.clone(), Box::new(v.diff(var)))),
                )),
                Box::new(Self::Pow(v.clone(), Box::new(Self::Const(2.0)))),
            ),
            // Power rule: (u^c)' = c * u^(c-1) * u'
            Self::Pow(u, exp) => {
                if let Self::Const(c) = **exp {
                    Self::Mul(
                        Box::new(Self::Mul(
                            Box::new(Self::Const(c)),
                            Box::new(Self::Pow(u.clone(), Box::new(Self::Const(c - 1.0)))),
                        )),
                        Box::new(u.diff(var)),
                    )
                } else {
                    // General u^v = exp(v * ln(u))
                    let ln_u = Self::Ln(u.clone());
                    let prod = Self::Mul(exp.clone(), Box::new(ln_u));
                    Self::Mul(Box::new(self.clone()), Box::new(prod.diff(var)))
                }
            }
            Self::Sin(u) => Self::Mul(
                Box::new(Self::Cos(u.clone())),
                Box::new(u.diff(var)),
            ),
            Self::Cos(u) => Self::Mul(
                Box::new(Self::Sub(
                    Box::new(Self::Const(0.0)),
                    Box::new(Self::Sin(u.clone())),
                )),
                Box::new(u.diff(var)),
            ),
            Self::Exp(u) => Self::Mul(
                Box::new(Self::Exp(u.clone())),
                Box::new(u.diff(var)),
            ),
            Self::Ln(u) => Self::Div(Box::new(u.diff(var)), u.clone()),
        };
        raw.simplify()
    }

    /// Higher-order symbolic derivative $\frac{\partial^n}{\partial \text{var}^n} f$.
    #[must_use]
    pub fn diff_n(&self, var: &str, order: usize) -> Self {
        let mut curr = self.clone();
        for _ in 0..order {
            curr = curr.diff(var);
        }
        curr
    }

    /// Recursive Algebraic Simplification (`simplify`):
    /// Constant folding, additive/multiplicative identities, power identities, and $\ln(e^x) \to x$.
    #[must_use]
    pub fn simplify(&self) -> Self {
        match self {
            Self::Const(c) => Self::Const(*c),
            Self::Var(v) => Self::Var(v.clone()),
            Self::Add(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Self::Const(ca), Self::Const(cb)) => Self::Const(ca + cb),
                    (Self::Const(c), _) if c.abs() == 0.0 => sb,
                    (_, Self::Const(c)) if c.abs() == 0.0 => sa,
                    _ if sa == sb => Self::Mul(Box::new(Self::Const(2.0)), Box::new(sa)),
                    _ => Self::Add(Box::new(sa), Box::new(sb)),
                }
            }
            Self::Sub(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Self::Const(ca), Self::Const(cb)) => Self::Const(ca - cb),
                    (_, Self::Const(c)) if c.abs() == 0.0 => sa,
                    _ if sa == sb => Self::Const(0.0),
                    _ => Self::Sub(Box::new(sa), Box::new(sb)),
                }
            }
            Self::Mul(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Self::Const(ca), Self::Const(cb)) => Self::Const(ca * cb),
                    (Self::Const(c), _) | (_, Self::Const(c)) if c.abs() == 0.0 => {
                        Self::Const(0.0)
                    }
                    (Self::Const(c), _) if (c - 1.0).abs() == 0.0 => sb,
                    (_, Self::Const(c)) if (c - 1.0).abs() == 0.0 => sa,
                    _ => Self::Mul(Box::new(sa), Box::new(sb)),
                }
            }
            Self::Div(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Self::Const(ca), Self::Const(cb)) if cb.abs() > 0.0 => Self::Const(ca / cb),
                    (Self::Const(c), _) if c.abs() == 0.0 => Self::Const(0.0),
                    (_, Self::Const(c)) if (c - 1.0).abs() == 0.0 => sa,
                    _ if sa == sb => Self::Const(1.0),
                    _ => Self::Div(Box::new(sa), Box::new(sb)),
                }
            }
            Self::Pow(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Self::Const(ca), Self::Const(cb)) => Self::Const(ca.powf(*cb)),
                    (_, Self::Const(c)) if c.abs() == 0.0 => Self::Const(1.0),
                    (_, Self::Const(c)) if (c - 1.0).abs() == 0.0 => sa,
                    _ => Self::Pow(Box::new(sa), Box::new(sb)),
                }
            }
            Self::Sin(a) => {
                let sa = a.simplify();
                if let Self::Const(c) = sa {
                    Self::Const(c.sin())
                } else {
                    Self::Sin(Box::new(sa))
                }
            }
            Self::Cos(a) => {
                let sa = a.simplify();
                if let Self::Const(c) = sa {
                    Self::Const(c.cos())
                } else {
                    Self::Cos(Box::new(sa))
                }
            }
            Self::Exp(a) => {
                let sa = a.simplify();
                match sa {
                    Self::Const(c) => Self::Const(c.exp()),
                    Self::Ln(inner) => *inner,
                    other => Self::Exp(Box::new(other)),
                }
            }
            Self::Ln(a) => {
                let sa = a.simplify();
                match sa {
                    Self::Const(c) => Self::Const(c.ln()),
                    Self::Exp(inner) => *inner,
                    other => Self::Ln(Box::new(other)),
                }
            }
        }
    }

    /// Symbolic Maclaurin/Taylor Series Expansion (`taylor(f, var, a, order)`):
    /// $\sum_{k=0}^{\text{order}} \frac{f^{(k)}(a)}{k!} (x - a)^k$.
    #[must_use]
    pub fn taylor(&self, var: &str, center: f64, order: usize) -> Self {
        let mut env = BTreeMap::new();
        env.insert(var.to_string(), center);

        let mut acc = Self::Const(0.0);
        let mut deriv = self.clone();
        let mut fact = 1.0;

        for k in 0..=order {
            if k > 0 {
                deriv = deriv.diff(var);
                fact *= k as f64;
            }
            let coeff = deriv.eval(&env) / fact;
            if coeff.abs() < 1e-15 {
                continue;
            }
            let dx = if center.abs() == 0.0 {
                Self::Var(var.to_string())
            } else {
                Self::Sub(
                    Box::new(Self::Var(var.to_string())),
                    Box::new(Self::Const(center)),
                )
            };
            let term = if k == 0 {
                Self::Const(coeff)
            } else {
                Self::Mul(
                    Box::new(Self::Const(coeff)),
                    Box::new(Self::Pow(Box::new(dx), Box::new(Self::Const(k as f64)))),
                )
            };
            acc = Self::Add(Box::new(acc), Box::new(term)).simplify();
        }
        acc
    }
}

impl fmt::Display for SymExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Const(c) => write!(f, "{c}"),
            Self::Var(v) => write!(f, "{v}"),
            Self::Add(a, b) => write!(f, "({a} + {b})"),
            Self::Sub(a, b) => write!(f, "({a} - {b})"),
            Self::Mul(a, b) => write!(f, "({a} * {b})"),
            Self::Div(a, b) => write!(f, "({a} / {b})"),
            Self::Pow(a, b) => write!(f, "({a}^{b})"),
            Self::Sin(a) => write!(f, "sin({a})"),
            Self::Cos(a) => write!(f, "cos({a})"),
            Self::Exp(a) => write!(f, "exp({a})"),
            Self::Ln(a) => write!(f, "ln({a})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_diff_simplify_and_taylor() {
        // f(x) = x^3 + sin(x) -> f'(x) = 3*x^2 + cos(x) -> f''(x) = 6*x - sin(x)
        let x = SymExpr::var("x");
        let fx = SymExpr::Add(
            Box::new(SymExpr::Pow(
                Box::new(x.clone()),
                Box::new(SymExpr::constant(3.0)),
            )),
            Box::new(SymExpr::Sin(Box::new(x.clone()))),
        );
        let df = fx.diff("x");
        let d2f = fx.diff_n("x", 2);

        let mut env = BTreeMap::new();
        env.insert("x".into(), 2.0);
        assert!((df.eval(&env) - (12.0 + 2.0_f64.cos())).abs() < 1e-12);
        assert!((d2f.eval(&env) - (12.0 - 2.0_f64.sin())).abs() < 1e-12);

        // Maclaurin series of exp(x) to order 4 evaluated at x = 0.5
        let exp_x = SymExpr::Exp(Box::new(x));
        let poly = exp_x.taylor("x", 0.0, 4);
        env.insert("x".into(), 0.5);
        assert!((poly.eval(&env) - 0.5_f64.exp()).abs() < 2e-3);
    }
}
