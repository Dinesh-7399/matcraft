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

pub mod bytecode;
pub mod hir;
pub mod oop;
pub mod symbolic;
pub mod type_checker;

pub use bytecode::{CompiledBytecode, OpCode};
pub use hir::{infer_program_units, StaticDiagnostic};
pub use oop::{
    parse_classdef, AccessModifier, ClassDef, ClassInstance, ClassRegistry, MethodDef, PropertyDef,
};
pub use symbolic::SymExpr;
pub use type_checker::{infer_expr, type_check_program, MType, TypeEnv, TypeWarning};

use num_core::{MemoryBudget, SiUnit, UnitError};
use std::collections::BTreeMap;
use std::sync::Arc;
use tensor::{Tensor, TensorError};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum LangError {
    #[error("Syntax error at token {pos}: {message}")]
    Syntax { pos: usize, message: String },
    #[error("Undefined variable or function '{0}'")]
    UndefinedSymbol(String),
    #[error("Unit error: {0}")]
    Unit(#[from] UnitError),
    #[error("Tensor error: {0}")]
    Tensor(#[from] TensorError),
    #[error("Execution fuel exhausted after {0} operations (possible infinite loop)")]
    FuelExhausted(u64),
    #[error("Recursion limit exceeded ({limit} frames)")]
    RecursionLimitExceeded { limit: usize },
    #[error("Builtin `{name}` error: {reason}")]
    BuiltinError { name: String, reason: String },
}

pub trait ExternalBuiltinProvider {
    fn call_external(
        &mut self,
        name: &str,
        args: &[Tensor<f64>],
        budget: &Arc<MemoryBudget>,
    ) -> Result<Option<Vec<Tensor<f64>>>, LangError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoopExternalProvider;
impl ExternalBuiltinProvider for NoopExternalProvider {
    fn call_external(
        &mut self,
        _name: &str,
        _args: &[Tensor<f64>],
        _budget: &Arc<MemoryBudget>,
    ) -> Result<Option<Vec<Tensor<f64>>>, LangError> {
        Ok(None)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Number(f64, Option<String>),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Backslash,
    Caret,
    DotStar,
    DotSlash,
    DotCaret,
    Transpose,
    Equal,
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    AndAnd,
    OrOr,
    Tilde,
    At,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Semicolon,
    Colon,
    Newline,
    KwFunction,
    KwReturn,
    KwIf,
    KwElseIf,
    KwElse,
    KwSwitch,
    KwCase,
    KwOtherwise,
    KwEnd,
    KwFor,
    KwWhile,
    KwBreak,
    KwContinue,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Literal(f64, SiUnit),
    Var(String),
    EndKeyword,
    ColonAll,
    Matrix(Vec<Vec<Expr>>),
    Range {
        start: Box<Expr>,
        step: Option<Box<Expr>>,
        stop: Box<Expr>,
    },
    UnaryNeg(Box<Expr>),
    LogicalNot(Box<Expr>),
    Transpose(Box<Expr>),
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    CallOrIndex {
        name: String,
        args: Vec<Expr>,
    },
    AnonFunc {
        params: Vec<String>,
        body: Box<Expr>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    MatMul,
    RightDiv,
    LeftDiv,
    ElemMul,
    ElemDiv,
    ElemPow,
    Pow,
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UserFunction {
    pub name: String,
    pub outputs: Vec<String>,
    pub params: Vec<String>,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Assign {
        target: String,
        expr: Expr,
        suppress_output: bool,
    },
    MultiAssign {
        targets: Vec<String>,
        expr: Expr,
        suppress_output: bool,
    },
    IndexedAssign {
        target: String,
        indices: Vec<Expr>,
        expr: Expr,
        suppress_output: bool,
    },
    Expr {
        expr: Expr,
        suppress_output: bool,
    },
    If {
        branches: Vec<(Expr, Vec<Stmt>)>,
        else_body: Vec<Stmt>,
    },
    Switch {
        expr: Expr,
        cases: Vec<(Expr, Vec<Stmt>)>,
        otherwise: Vec<Stmt>,
    },
    For {
        var: String,
        range: Expr,
        body: Vec<Stmt>,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
    },
    FunctionDef(UserFunction),
    Return,
    Break,
    Continue,
}

pub fn tokenize(src: &str) -> Result<Vec<Token>, LangError> {
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0usize;
    let mut tokens = Vec::new();

    while i < chars.len() {
        let ch = chars.get(i).copied().unwrap_or(' ');
        if ch == ' ' || ch == '\t' || ch == '\r' {
            i = i.saturating_add(1);
            continue;
        }
        if ch == '%' {
            while i < chars.len() && chars.get(i).copied() != Some('\n') {
                i = i.saturating_add(1);
            }
            continue;
        }
        if ch == '\n' {
            tokens.push(Token::Newline);
            i = i.saturating_add(1);
            continue;
        }
        if ch.is_ascii_digit()
            || (ch == '.'
                && chars
                    .get(i.saturating_add(1))
                    .is_some_and(char::is_ascii_digit))
        {
            let start = i;
            while i < chars.len()
                && chars
                    .get(i)
                    .is_some_and(|c| c.is_ascii_digit() || *c == '.')
            {
                i = i.saturating_add(1);
            }
            if chars
                .get(i)
                .is_some_and(|&c| (c == 'e' || c == 'E') && i > start)
            {
                i = i.saturating_add(1);
                if chars.get(i).is_some_and(|&c| c == '+' || c == '-') {
                    i = i.saturating_add(1);
                }
                while i < chars.len() && chars.get(i).is_some_and(char::is_ascii_digit) {
                    i = i.saturating_add(1);
                }
            }
            let num_str: String = chars.get(start..i).unwrap_or(&[]).iter().collect();
            let val: f64 = num_str.parse().map_err(|_| LangError::Syntax {
                pos: start,
                message: format!("Invalid float literal '{num_str}'"),
            })?;

            let unit_suffix = if chars.get(i) == Some(&'_')
                && chars.get(i.saturating_add(1)).is_some_and(char::is_ascii_alphabetic)
            {
                i = i.saturating_add(1);
                let u_start = i;
                while i < chars.len() {
                    let c = chars.get(i).copied().unwrap_or('\0');
                    if c.is_ascii_alphabetic() {
                        i = i.saturating_add(1);
                    } else if c == '^' {
                        let next_c = chars.get(i.saturating_add(1)).copied();
                        let next2_c = chars.get(i.saturating_add(2)).copied();
                        if next_c.is_some_and(|ch| ch.is_ascii_digit()) {
                            i = i.saturating_add(1);
                            while i < chars.len() && chars.get(i).is_some_and(char::is_ascii_digit) {
                                i = i.saturating_add(1);
                            }
                        } else if next_c == Some('-') && next2_c.is_some_and(|ch| ch.is_ascii_digit()) {
                            i = i.saturating_add(2);
                            while i < chars.len() && chars.get(i).is_some_and(char::is_ascii_digit) {
                                i = i.saturating_add(1);
                            }
                        } else {
                            break;
                        }
                    } else if (c == '*' || c == '/')
                        && chars.get(i.saturating_add(1)).is_some_and(char::is_ascii_alphabetic)
                    {
                        i = i.saturating_add(1);
                    } else {
                        break;
                    }
                }
                let u_str: String = chars.get(u_start..i).unwrap_or(&[]).iter().collect();
                Some(u_str)
            } else {
                None
            };
            tokens.push(Token::Number(val, unit_suffix));
            continue;
        }

        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = i;
            while i < chars.len()
                && chars
                    .get(i)
                    .is_some_and(|&c| c.is_ascii_alphanumeric() || c == '_')
            {
                i = i.saturating_add(1);
            }
            let ident: String = chars.get(start..i).unwrap_or(&[]).iter().collect();
            let tok = match ident.as_str() {
                "function" => Token::KwFunction,
                "return" => Token::KwReturn,
                "if" => Token::KwIf,
                "elseif" => Token::KwElseIf,
                "else" => Token::KwElse,
                "switch" => Token::KwSwitch,
                "case" => Token::KwCase,
                "otherwise" => Token::KwOtherwise,
                "end" => Token::KwEnd,
                "for" => Token::KwFor,
                "while" => Token::KwWhile,
                "break" => Token::KwBreak,
                "continue" => Token::KwContinue,
                _ => Token::Ident(ident),
            };
            tokens.push(tok);
            continue;
        }

        let next_ch = chars.get(i.saturating_add(1)).copied();
        match (ch, next_ch) {
            ('.', Some('*')) => {
                tokens.push(Token::DotStar);
                i = i.saturating_add(2);
                continue;
            }
            ('.', Some('/')) => {
                tokens.push(Token::DotSlash);
                i = i.saturating_add(2);
                continue;
            }
            ('.', Some('^')) => {
                tokens.push(Token::DotCaret);
                i = i.saturating_add(2);
                continue;
            }
            ('.', Some('\'')) => {
                tokens.push(Token::Transpose);
                i = i.saturating_add(2);
                continue;
            }
            ('=', Some('=')) => {
                tokens.push(Token::EqEq);
                i = i.saturating_add(2);
                continue;
            }
            ('~', Some('=')) => {
                tokens.push(Token::NotEq);
                i = i.saturating_add(2);
                continue;
            }
            ('<', Some('=')) => {
                tokens.push(Token::Le);
                i = i.saturating_add(2);
                continue;
            }
            ('>', Some('=')) => {
                tokens.push(Token::Ge);
                i = i.saturating_add(2);
                continue;
            }
            ('&', Some('&')) => {
                tokens.push(Token::AndAnd);
                i = i.saturating_add(2);
                continue;
            }
            ('|', Some('|')) => {
                tokens.push(Token::OrOr);
                i = i.saturating_add(2);
                continue;
            }
            _ => {}
        }

        let single = match ch {
            '+' => Token::Plus,
            '-' => Token::Minus,
            '*' => Token::Star,
            '/' => Token::Slash,
            '\\' => Token::Backslash,
            '^' => Token::Caret,
            '\'' => Token::Transpose,
            '=' => Token::Equal,
            '<' => Token::Lt,
            '>' => Token::Gt,
            '~' => Token::Tilde,
            '@' => Token::At,
            '&' => Token::AndAnd,
            '|' => Token::OrOr,
            '(' => Token::LParen,
            ')' => Token::RParen,
            '[' => Token::LBracket,
            ']' => Token::RBracket,
            ',' => Token::Comma,
            ';' => Token::Semicolon,
            ':' => Token::Colon,
            other => {
                return Err(LangError::Syntax {
                    pos: i,
                    message: format!("Unexpected character '{other}'"),
                })
            }
        };
        tokens.push(single);
        i = i.saturating_add(1);
    }

    Ok(tokens)
}

#[derive(Debug)]
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    inside_index_depth: usize,
    parse_depth: usize,
}

impl Parser {
    pub const MAX_PARSE_DEPTH: usize = 256;

    #[must_use]
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            inside_index_depth: 0,
            parse_depth: 0,
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next_tok(&mut self) -> Option<Token> {
        let tok = self.tokens.get(self.pos).cloned();
        if tok.is_some() {
            self.pos = self.pos.saturating_add(1);
        }
        tok
    }

    fn skip_separators(&mut self) {
        while matches!(
            self.peek(),
            Some(Token::Newline | Token::Semicolon | Token::Comma)
        ) {
            self.pos = self.pos.saturating_add(1);
        }
    }

    pub fn parse_program(&mut self) -> Result<Vec<Stmt>, LangError> {
        let mut stmts = Vec::new();
        self.skip_separators();
        while self.peek().is_some() {
            stmts.push(self.parse_stmt()?);
            self.skip_separators();
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, LangError> {
        match self.peek() {
            Some(Token::KwFunction) => self.parse_function_def(),
            Some(Token::KwReturn) => {
                self.next_tok();
                let _ = self.consume_suppress();
                Ok(Stmt::Return)
            }
            Some(Token::KwBreak) => {
                self.next_tok();
                let _ = self.consume_suppress();
                Ok(Stmt::Break)
            }
            Some(Token::KwContinue) => {
                self.next_tok();
                let _ = self.consume_suppress();
                Ok(Stmt::Continue)
            }
            Some(Token::KwIf) => self.parse_if(),
            Some(Token::KwSwitch) => self.parse_switch(),
            Some(Token::KwFor) => self.parse_for(),
            Some(Token::KwWhile) => self.parse_while(),
            Some(Token::LBracket) => {
                // Check for Multi-Assignment `[U, S, V] = svd(A)`
                let saved = self.pos;
                if let Ok(targets) = self.try_parse_multi_assign_lhs() {
                    let expr = self.parse_expr()?;
                    let suppress = self.consume_suppress();
                    return Ok(Stmt::MultiAssign {
                        targets,
                        expr,
                        suppress_output: suppress,
                    });
                }
                self.pos = saved;
                let expr = self.parse_expr()?;
                let suppress = self.consume_suppress();
                Ok(Stmt::Expr {
                    expr,
                    suppress_output: suppress,
                })
            }
            Some(Token::Ident(_)) => {
                let saved = self.pos;
                if let Some(Token::Ident(name)) = self.next_tok() {
                    if self.peek() == Some(&Token::Equal) {
                        self.next_tok();
                        let expr = self.parse_expr()?;
                        let suppress = self.consume_suppress();
                        return Ok(Stmt::Assign {
                            target: name,
                            expr,
                            suppress_output: suppress,
                        });
                    } else if self.peek() == Some(&Token::LParen) {
                        self.next_tok();
                        self.inside_index_depth = self.inside_index_depth.saturating_add(1);
                        let mut indices = Vec::new();
                        if self.peek() != Some(&Token::RParen) {
                            loop {
                                indices.push(self.parse_expr()?);
                                if self.peek() == Some(&Token::Comma) {
                                    self.next_tok();
                                } else {
                                    break;
                                }
                            }
                        }
                        self.inside_index_depth = self.inside_index_depth.saturating_sub(1);
                        if self.peek() == Some(&Token::RParen) {
                            self.next_tok();
                            if self.peek() == Some(&Token::Equal) {
                                self.next_tok();
                                let expr = self.parse_expr()?;
                                let suppress = self.consume_suppress();
                                return Ok(Stmt::IndexedAssign {
                                    target: name,
                                    indices,
                                    expr,
                                    suppress_output: suppress,
                                });
                            }
                        }
                    }
                }
                self.pos = saved;
                let expr = self.parse_expr()?;
                let suppress = self.consume_suppress();
                Ok(Stmt::Expr {
                    expr,
                    suppress_output: suppress,
                })
            }
            _ => {
                let expr = self.parse_expr()?;
                let suppress = self.consume_suppress();
                Ok(Stmt::Expr {
                    expr,
                    suppress_output: suppress,
                })
            }
        }
    }

    fn try_parse_multi_assign_lhs(&mut self) -> Result<Vec<String>, LangError> {
        if self.next_tok() != Some(Token::LBracket) {
            return Err(LangError::Syntax {
                pos: self.pos,
                message: "Not `[`".into(),
            });
        }
        let mut targets = Vec::new();
        while self.peek().is_some() && self.peek() != Some(&Token::RBracket) {
            if self.peek() == Some(&Token::Comma) {
                self.next_tok();
                continue;
            }
            if let Some(Token::Ident(id)) = self.next_tok() {
                targets.push(id);
            } else {
                return Err(LangError::Syntax {
                    pos: self.pos,
                    message: "Not identifier list".into(),
                });
            }
        }
        if self.next_tok() != Some(Token::RBracket) || self.next_tok() != Some(Token::Equal) {
            return Err(LangError::Syntax {
                pos: self.pos,
                message: "Not multi-assign".into(),
            });
        }
        Ok(targets)
    }

    fn parse_function_def(&mut self) -> Result<Stmt, LangError> {
        self.next_tok(); // KwFunction
        let mut outputs = Vec::new();
        let fn_name: String;

        if self.peek() == Some(&Token::LBracket) {
            self.next_tok();
            while self.peek().is_some() && self.peek() != Some(&Token::RBracket) {
                if self.peek() == Some(&Token::Comma) {
                    self.next_tok();
                    continue;
                }
                let Some(Token::Ident(out_id)) = self.next_tok() else {
                    return Err(LangError::Syntax {
                        pos: self.pos,
                        message: "Expected output identifier in function header".into(),
                    });
                };
                outputs.push(out_id);
            }
            if self.next_tok() != Some(Token::RBracket) || self.next_tok() != Some(Token::Equal) {
                return Err(LangError::Syntax {
                    pos: self.pos,
                    message: "Expected `] =` in function definition".into(),
                });
            }
            let Some(Token::Ident(name)) = self.next_tok() else {
                return Err(LangError::Syntax {
                    pos: self.pos,
                    message: "Expected function name after `=`".into(),
                });
            };
            fn_name = name;
        } else {
            let Some(Token::Ident(first_id)) = self.next_tok() else {
                return Err(LangError::Syntax {
                    pos: self.pos,
                    message: "Expected function name or output variable".into(),
                });
            };
            if self.peek() == Some(&Token::Equal) {
                self.next_tok();
                outputs.push(first_id);
                let Some(Token::Ident(name)) = self.next_tok() else {
                    return Err(LangError::Syntax {
                        pos: self.pos,
                        message: "Expected function name after `=`".into(),
                    });
                };
                fn_name = name;
            } else {
                fn_name = first_id;
            }
        }

        let mut params = Vec::new();
        if self.peek() == Some(&Token::LParen) {
            self.next_tok();
            while self.peek().is_some() && self.peek() != Some(&Token::RParen) {
                if self.peek() == Some(&Token::Comma) {
                    self.next_tok();
                    continue;
                }
                let Some(Token::Ident(p)) = self.next_tok() else {
                    return Err(LangError::Syntax {
                        pos: self.pos,
                        message: "Expected parameter identifier".into(),
                    });
                };
                params.push(p);
            }
            if self.next_tok() != Some(Token::RParen) {
                return Err(LangError::Syntax {
                    pos: self.pos,
                    message: "Expected `)` after function parameters".into(),
                });
            }
        }

        self.skip_separators();
        let mut body = Vec::new();
        while !matches!(self.peek(), Some(Token::KwEnd) | None) {
            body.push(self.parse_stmt()?);
            self.skip_separators();
        }
        if self.next_tok() != Some(Token::KwEnd) {
            return Err(LangError::Syntax {
                pos: self.pos,
                message: "Expected `end` to close `function`".into(),
            });
        }
        Ok(Stmt::FunctionDef(UserFunction {
            name: fn_name,
            outputs,
            params,
            body,
        }))
    }

    fn consume_suppress(&mut self) -> bool {
        if self.peek() == Some(&Token::Semicolon) {
            self.next_tok();
            true
        } else {
            false
        }
    }

    fn parse_if(&mut self) -> Result<Stmt, LangError> {
        self.next_tok(); // KwIf
        let first_cond = self.parse_expr()?;
        self.skip_separators();
        let mut first_body = Vec::new();
        while !matches!(
            self.peek(),
            Some(Token::KwElseIf | Token::KwElse | Token::KwEnd) | None
        ) {
            first_body.push(self.parse_stmt()?);
            self.skip_separators();
        }
        let mut branches = vec![(first_cond, first_body)];

        while self.peek() == Some(&Token::KwElseIf) {
            self.next_tok();
            let c = self.parse_expr()?;
            self.skip_separators();
            let mut b = Vec::new();
            while !matches!(
                self.peek(),
                Some(Token::KwElseIf | Token::KwElse | Token::KwEnd) | None
            ) {
                b.push(self.parse_stmt()?);
                self.skip_separators();
            }
            branches.push((c, b));
        }

        let mut else_body = Vec::new();
        if self.peek() == Some(&Token::KwElse) {
            self.next_tok();
            self.skip_separators();
            while !matches!(self.peek(), Some(Token::KwEnd) | None) {
                else_body.push(self.parse_stmt()?);
                self.skip_separators();
            }
        }
        if self.next_tok() != Some(Token::KwEnd) {
            return Err(LangError::Syntax {
                pos: self.pos,
                message: "Expected `end` to close `if`".into(),
            });
        }
        Ok(Stmt::If {
            branches,
            else_body,
        })
    }

    fn parse_switch(&mut self) -> Result<Stmt, LangError> {
        self.next_tok(); // KwSwitch
        let expr = self.parse_expr()?;
        self.skip_separators();
        let mut cases = Vec::new();
        let mut otherwise = Vec::new();

        while self.peek() == Some(&Token::KwCase) {
            self.next_tok();
            let c_expr = self.parse_expr()?;
            self.skip_separators();
            let mut body = Vec::new();
            while !matches!(
                self.peek(),
                Some(Token::KwCase | Token::KwOtherwise | Token::KwEnd) | None
            ) {
                body.push(self.parse_stmt()?);
                self.skip_separators();
            }
            cases.push((c_expr, body));
        }
        if self.peek() == Some(&Token::KwOtherwise) {
            self.next_tok();
            self.skip_separators();
            while !matches!(self.peek(), Some(Token::KwEnd) | None) {
                otherwise.push(self.parse_stmt()?);
                self.skip_separators();
            }
        }
        if self.next_tok() != Some(Token::KwEnd) {
            return Err(LangError::Syntax {
                pos: self.pos,
                message: "Expected `end` to close `switch`".into(),
            });
        }
        Ok(Stmt::Switch {
            expr,
            cases,
            otherwise,
        })
    }

    fn parse_for(&mut self) -> Result<Stmt, LangError> {
        self.next_tok();
        let Some(Token::Ident(var)) = self.next_tok() else {
            return Err(LangError::Syntax {
                pos: self.pos,
                message: "Expected loop variable identifier after `for`".into(),
            });
        };
        if self.next_tok() != Some(Token::Equal) {
            return Err(LangError::Syntax {
                pos: self.pos,
                message: "Expected `=` in `for` loop header".into(),
            });
        }
        let range = self.parse_expr()?;
        self.skip_separators();
        let mut body = Vec::new();
        while !matches!(self.peek(), Some(Token::KwEnd) | None) {
            body.push(self.parse_stmt()?);
            self.skip_separators();
        }
        if self.next_tok() != Some(Token::KwEnd) {
            return Err(LangError::Syntax {
                pos: self.pos,
                message: "Expected `end` to close `for` loop".into(),
            });
        }
        Ok(Stmt::For { var, range, body })
    }

    fn parse_while(&mut self) -> Result<Stmt, LangError> {
        self.next_tok();
        let cond = self.parse_expr()?;
        self.skip_separators();
        let mut body = Vec::new();
        while !matches!(self.peek(), Some(Token::KwEnd) | None) {
            body.push(self.parse_stmt()?);
            self.skip_separators();
        }
        if self.next_tok() != Some(Token::KwEnd) {
            return Err(LangError::Syntax {
                pos: self.pos,
                message: "Expected `end` to close `while` loop".into(),
            });
        }
        Ok(Stmt::While { cond, body })
    }

    pub fn parse_expr(&mut self) -> Result<Expr, LangError> {
        if self.parse_depth >= Self::MAX_PARSE_DEPTH {
            return Err(LangError::RecursionLimitExceeded {
                limit: Self::MAX_PARSE_DEPTH,
            });
        }
        self.parse_depth = self.parse_depth.saturating_add(1);
        let res = self.parse_logical_or();
        self.parse_depth = self.parse_depth.saturating_sub(1);
        res
    }

    fn parse_logical_or(&mut self) -> Result<Expr, LangError> {
        let mut lhs = self.parse_logical_and()?;
        while self.peek() == Some(&Token::OrOr) {
            self.next_tok();
            let rhs = self.parse_logical_and()?;
            lhs = Expr::Binary {
                op: BinOp::Or,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_logical_and(&mut self) -> Result<Expr, LangError> {
        let mut lhs = self.parse_comparison()?;
        while self.peek() == Some(&Token::AndAnd) {
            self.next_tok();
            let rhs = self.parse_comparison()?;
            lhs = Expr::Binary {
                op: BinOp::And,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_comparison(&mut self) -> Result<Expr, LangError> {
        let mut lhs = self.parse_range()?;
        while let Some(tok) = self.peek() {
            let op = match tok {
                Token::EqEq => BinOp::EqEq,
                Token::NotEq => BinOp::NotEq,
                Token::Lt => BinOp::Lt,
                Token::Le => BinOp::Le,
                Token::Gt => BinOp::Gt,
                Token::Ge => BinOp::Ge,
                _ => break,
            };
            self.next_tok();
            let rhs = self.parse_range()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_range(&mut self) -> Result<Expr, LangError> {
        if self.peek() == Some(&Token::Colon) {
            self.next_tok();
            return Ok(Expr::ColonAll);
        }
        let first = self.parse_add_sub()?;
        if self.peek() == Some(&Token::Colon) {
            self.next_tok();
            let second = self.parse_add_sub()?;
            if self.peek() == Some(&Token::Colon) {
                self.next_tok();
                let third = self.parse_add_sub()?;
                return Ok(Expr::Range {
                    start: Box::new(first),
                    step: Some(Box::new(second)),
                    stop: Box::new(third),
                });
            }
            return Ok(Expr::Range {
                start: Box::new(first),
                step: None,
                stop: Box::new(second),
            });
        }
        Ok(first)
    }

    fn parse_add_sub(&mut self) -> Result<Expr, LangError> {
        let mut lhs = self.parse_mul_div()?;
        while let Some(tok) = self.peek() {
            let op = match tok {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => break,
            };
            self.next_tok();
            let rhs = self.parse_mul_div()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_mul_div(&mut self) -> Result<Expr, LangError> {
        let mut lhs = self.parse_unary()?;
        while let Some(tok) = self.peek() {
            let op = match tok {
                Token::Star => BinOp::MatMul,
                Token::Slash => BinOp::RightDiv,
                Token::Backslash => BinOp::LeftDiv,
                Token::DotStar => BinOp::ElemMul,
                Token::DotSlash => BinOp::ElemDiv,
                _ => break,
            };
            self.next_tok();
            let rhs = self.parse_unary()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, LangError> {
        if self.peek() == Some(&Token::Minus) {
            self.next_tok();
            let inner = self.parse_unary()?;
            return Ok(Expr::UnaryNeg(Box::new(inner)));
        }
        if self.peek() == Some(&Token::Tilde) {
            self.next_tok();
            let inner = self.parse_unary()?;
            return Ok(Expr::LogicalNot(Box::new(inner)));
        }
        if self.peek() == Some(&Token::Plus) {
            self.next_tok();
            return self.parse_unary();
        }
        self.parse_power()
    }

    fn parse_power(&mut self) -> Result<Expr, LangError> {
        let mut base = self.parse_postfix()?;
        while let Some(tok) = self.peek() {
            let op = match tok {
                Token::Caret => BinOp::Pow,
                Token::DotCaret => BinOp::ElemPow,
                _ => break,
            };
            self.next_tok();
            let exp = self.parse_unary()?;
            base = Expr::Binary {
                op,
                lhs: Box::new(base),
                rhs: Box::new(exp),
            };
        }
        Ok(base)
    }

    fn parse_postfix(&mut self) -> Result<Expr, LangError> {
        let mut expr = self.parse_primary()?;
        while self.peek() == Some(&Token::Transpose) {
            self.next_tok();
            expr = Expr::Transpose(Box::new(expr));
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, LangError> {
        match self.next_tok() {
            Some(Token::Number(val, unit_opt)) => {
                let unit = match unit_opt {
                    Some(u_str) => SiUnit::parse(&u_str)?,
                    None => SiUnit::DIMENSIONLESS,
                };
                Ok(Expr::Literal(val, unit))
            }
            Some(Token::KwEnd) if self.inside_index_depth > 0 => Ok(Expr::EndKeyword),
            Some(Token::At) => {
                // Anonymous function `@(x, y) expr`
                if self.next_tok() != Some(Token::LParen) {
                    return Err(LangError::Syntax {
                        pos: self.pos,
                        message: "Expected `(` after `@` in anonymous function".into(),
                    });
                }
                let mut params = Vec::new();
                while self.peek().is_some() && self.peek() != Some(&Token::RParen) {
                    if self.peek() == Some(&Token::Comma) {
                        self.next_tok();
                        continue;
                    }
                    let Some(Token::Ident(p)) = self.next_tok() else {
                        return Err(LangError::Syntax {
                            pos: self.pos,
                            message: "Expected parameter name in `@(...)`".into(),
                        });
                    };
                    params.push(p);
                }
                if self.next_tok() != Some(Token::RParen) {
                    return Err(LangError::Syntax {
                        pos: self.pos,
                        message: "Expected `)` in anonymous function".into(),
                    });
                }
                let body = self.parse_expr()?;
                Ok(Expr::AnonFunc {
                    params,
                    body: Box::new(body),
                })
            }
            Some(Token::Ident(name)) => {
                if self.peek() == Some(&Token::LParen) {
                    self.next_tok();
                    self.inside_index_depth = self.inside_index_depth.saturating_add(1);
                    let mut args = Vec::new();
                    if self.peek() != Some(&Token::RParen) {
                        loop {
                            args.push(self.parse_expr()?);
                            if self.peek() == Some(&Token::Comma) {
                                self.next_tok();
                            } else {
                                break;
                            }
                        }
                    }
                    self.inside_index_depth = self.inside_index_depth.saturating_sub(1);
                    if self.next_tok() != Some(Token::RParen) {
                        return Err(LangError::Syntax {
                            pos: self.pos,
                            message: "Expected `)` after argument list".into(),
                        });
                    }
                    Ok(Expr::CallOrIndex { name, args })
                } else {
                    Ok(Expr::Var(name))
                }
            }
            Some(Token::LParen) => {
                let inner = self.parse_expr()?;
                if self.next_tok() != Some(Token::RParen) {
                    return Err(LangError::Syntax {
                        pos: self.pos,
                        message: "Expected closing `)`".into(),
                    });
                }
                Ok(inner)
            }
            Some(Token::LBracket) => self.parse_matrix_literal(),
            other => Err(LangError::Syntax {
                pos: self.pos,
                message: format!("Unexpected token {other:?} in expression"),
            }),
        }
    }

    fn parse_matrix_literal(&mut self) -> Result<Expr, LangError> {
        let mut rows = Vec::new();
        let mut current_row = Vec::new();

        while self.peek().is_some() && self.peek() != Some(&Token::RBracket) {
            if matches!(self.peek(), Some(Token::Semicolon | Token::Newline)) {
                self.next_tok();
                if !current_row.is_empty() {
                    rows.push(current_row);
                    current_row = Vec::new();
                }
                continue;
            }
            if self.peek() == Some(&Token::Comma) {
                self.next_tok();
                continue;
            }
            current_row.push(self.parse_expr()?);
        }
        if !current_row.is_empty() {
            rows.push(current_row);
        }
        if self.next_tok() != Some(Token::RBracket) {
            return Err(LangError::Syntax {
                pos: self.pos,
                message: "Expected closing `]` for matrix literal".into(),
            });
        }
        Ok(Expr::Matrix(rows))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlowSignal {
    None,
    Break,
    Continue,
    Return,
}

/// Fuel-metered `.m` Script & Function Evaluator.
#[derive(Debug)]
pub struct Interpreter {
    pub vars: BTreeMap<String, Tensor<f64>>,
    pub functions: BTreeMap<String, UserFunction>,
    pub anon_funcs: BTreeMap<String, (Vec<String>, Expr)>,
    pub budget: Arc<MemoryBudget>,
    pub fuel_remaining: u64,
    pub max_fuel: u64,
    pub call_depth: usize,
    active_end_len: Option<usize>,
}

impl Interpreter {
    pub const DEFAULT_FUEL: u64 = 500_000;
    pub const MAX_CALL_DEPTH: usize = 32;

    #[must_use]
    pub fn new(budget: Arc<MemoryBudget>) -> Self {
        let mut vars = BTreeMap::new();
        if let Ok(pi_t) = Tensor::scalar(std::f64::consts::PI, SiUnit::DIMENSIONLESS, &budget) {
            vars.insert("pi".to_string(), pi_t);
        }
        Self {
            vars,
            functions: BTreeMap::new(),
            anon_funcs: BTreeMap::new(),
            budget,
            fuel_remaining: Self::DEFAULT_FUEL,
            max_fuel: Self::DEFAULT_FUEL,
            call_depth: 0,
            active_end_len: None,
        }
    }

    fn tick_fuel(&mut self) -> Result<(), LangError> {
        if self.fuel_remaining == 0 {
            return Err(LangError::FuelExhausted(self.max_fuel));
        }
        self.fuel_remaining = self.fuel_remaining.saturating_sub(1);
        Ok(())
    }

    pub fn eval_script<P: ExternalBuiltinProvider>(
        &mut self,
        source: &str,
        external: &mut P,
    ) -> Result<Option<Tensor<f64>>, LangError> {
        self.fuel_remaining = self.max_fuel;
        let tokens = tokenize(source)?;
        let mut parser = Parser::new(tokens);
        let stmts = parser.parse_program()?;

        // Register all top-level function definitions first
        for stmt in &stmts {
            if let Stmt::FunctionDef(f) = stmt {
                self.functions.insert(f.name.clone(), f.clone());
            }
        }

        let mut last_val = None;
        for stmt in &stmts {
            if matches!(stmt, Stmt::FunctionDef(_)) {
                continue;
            }
            let (val, sig) = self.exec_stmt(stmt, external)?;
            if val.is_some() {
                last_val = val;
            }
            if sig != FlowSignal::None {
                break;
            }
        }
        Ok(last_val)
    }

    fn exec_stmt<P: ExternalBuiltinProvider>(
        &mut self,
        stmt: &Stmt,
        external: &mut P,
    ) -> Result<(Option<Tensor<f64>>, FlowSignal), LangError> {
        self.tick_fuel()?;
        match stmt {
            Stmt::FunctionDef(f) => {
                self.functions.insert(f.name.clone(), f.clone());
                Ok((None, FlowSignal::None))
            }
            Stmt::Return => Ok((None, FlowSignal::Return)),
            Stmt::Break => Ok((None, FlowSignal::Break)),
            Stmt::Continue => Ok((None, FlowSignal::Continue)),
            Stmt::Assign { target, expr, .. } => {
                if let Expr::AnonFunc { params, body } = expr {
                    self.anon_funcs
                        .insert(target.clone(), (params.clone(), (**body).clone()));
                    let handle_scalar =
                        Tensor::scalar(0.0, SiUnit::DIMENSIONLESS, &self.budget)?;
                    self.vars.insert(target.clone(), handle_scalar.clone());
                    return Ok((Some(handle_scalar), FlowSignal::None));
                }
                self.anon_funcs.remove(target);
                let val = self.eval_expr(expr, external)?;
                self.vars.insert(target.clone(), val.clone());
                Ok((Some(val), FlowSignal::None))
            }
            Stmt::MultiAssign { targets, expr, .. } => {
                let Expr::CallOrIndex { name, args } = expr else {
                    return Err(LangError::Syntax {
                        pos: 0,
                        message: "Right-hand side of multi-assignment must be a function call"
                            .into(),
                    });
                };
                for t in targets {
                    self.anon_funcs.remove(t);
                }
                let eval_args: Vec<Tensor<f64>> = args
                    .iter()
                    .map(|a| self.eval_expr(a, external))
                    .collect::<Result<Vec<_>, _>>()?;
                let outputs = self.call_multi_output_function(name, &eval_args, external)?;
                let mut last = None;
                for (t_name, out_tensor) in targets.iter().zip(outputs) {
                    self.vars.insert(t_name.clone(), out_tensor.clone());
                    last = Some(out_tensor);
                }
                Ok((last, FlowSignal::None))
            }
            Stmt::IndexedAssign {
                target,
                indices,
                expr,
                ..
            } => {
                let rhs = self.eval_expr(expr, external)?;
                let scalar_val = rhs.get(0, 0)?;
                let (rows, cols, numel) = {
                    let mat = self
                        .vars
                        .get(target)
                        .ok_or_else(|| LangError::UndefinedSymbol(target.clone()))?;
                    (mat.rows(), mat.cols(), mat.numel())
                };
                if indices.len() == 2 {
                    let r_idx = self.resolve_index_list(
                        indices.first().ok_or_else(|| LangError::UndefinedSymbol(target.clone()))?,
                        rows,
                        external,
                    )?;
                    let c_idx = self.resolve_index_list(
                        indices.get(1).ok_or_else(|| LangError::UndefinedSymbol(target.clone()))?,
                        cols,
                        external,
                    )?;
                    let needed = r_idx.len().saturating_mul(c_idx.len());
                    if rhs.numel() != 1 && rhs.numel() != needed {
                        return Err(LangError::Tensor(TensorError::ShapeMismatch {
                            op: "indexed assignment",
                            lhs: vec![r_idx.len(), c_idx.len()],
                            rhs: rhs.shape().to_vec(),
                        }));
                    }
                    let mat = self
                        .vars
                        .get_mut(target)
                        .ok_or_else(|| LangError::UndefinedSymbol(target.clone()))?;
                    for (c_sub, &c0) in c_idx.iter().enumerate() {
                        for (r_sub, &r0) in r_idx.iter().enumerate() {
                            let val = if rhs.numel() == 1 {
                                scalar_val
                            } else if rhs.shape() == [r_idx.len(), c_idx.len()] {
                                rhs.get(r_sub, c_sub)?
                            } else {
                                let lin = c_sub.saturating_mul(r_idx.len()).saturating_add(r_sub);
                                rhs.get_linear(lin)?
                            };
                            mat.set(r0, c0, val)?;
                        }
                    }
                } else if indices.len() == 1 {
                    let first_expr = indices
                        .first()
                        .ok_or_else(|| LangError::UndefinedSymbol(target.clone()))?;
                    self.active_end_len = Some(numel);
                    let idx_tensor = self.eval_expr(first_expr, external)?;
                    self.active_end_len = None;
                    if idx_tensor.shape() == [rows, cols]
                        && idx_tensor
                            .to_column_major_vec()?
                            .iter()
                            .all(|&v| v == 0.0 || v == 1.0)
                    {
                        // Logical mask assignment `A(A < 0) = 0`
                        let mask_count = idx_tensor
                            .to_column_major_vec()?
                            .iter()
                            .filter(|&&v| v != 0.0)
                            .count();
                        if rhs.numel() != 1 && rhs.numel() != mask_count {
                            return Err(LangError::Tensor(TensorError::ShapeMismatch {
                                op: "logical mask assignment",
                                lhs: vec![mask_count],
                                rhs: vec![rhs.numel()],
                            }));
                        }
                        let mat = self
                            .vars
                            .get_mut(target)
                            .ok_or_else(|| LangError::UndefinedSymbol(target.clone()))?;
                        let mut sub_i = 0usize;
                        for c in 0..cols {
                            for r in 0..rows {
                                if idx_tensor.get(r, c)? != 0.0 {
                                    let val = if rhs.numel() == 1 {
                                        scalar_val
                                    } else {
                                        rhs.get_linear(sub_i)?
                                    };
                                    mat.set(r, c, val)?;
                                    sub_i = sub_i.saturating_add(1);
                                }
                            }
                        }
                    } else {
                        let total_indices = idx_tensor.numel();
                        if rhs.numel() != 1 && rhs.numel() != total_indices {
                            return Err(LangError::Tensor(TensorError::ShapeMismatch {
                                op: "linear indexed assignment",
                                lhs: vec![total_indices],
                                rhs: vec![rhs.numel()],
                            }));
                        }
                        let mat = self
                            .vars
                            .get_mut(target)
                            .ok_or_else(|| LangError::UndefinedSymbol(target.clone()))?;
                        for i in 0..total_indices {
                            let idx1 = idx_tensor.get_linear(i)? as usize;
                            if idx1 == 0 {
                                return Err(LangError::Tensor(TensorError::IndexOutOfBounds {
                                    index: vec![0],
                                    shape: vec![rows, cols],
                                }));
                            }
                            let val = if rhs.numel() == 1 {
                                scalar_val
                            } else {
                                rhs.get_linear(i)?
                            };
                            mat.set_linear(idx1 - 1, val)?;
                        }
                    }
                }
                Ok((self.vars.get(target).cloned(), FlowSignal::None))
            }
            Stmt::Expr { expr, .. } => {
                let val = self.eval_expr(expr, external)?;
                self.vars.insert("ans".to_string(), val.clone());
                Ok((Some(val), FlowSignal::None))
            }
            Stmt::If {
                branches,
                else_body,
            } => {
                for (cond, body) in branches {
                    let c = self.eval_expr(cond, external)?;
                    if !c.is_empty() && c.get(0, 0)? != 0.0 {
                        return self.exec_block(body, external);
                    }
                }
                self.exec_block(else_body, external)
            }
            Stmt::Switch {
                expr,
                cases,
                otherwise,
            } => {
                let target_v = self.eval_expr(expr, external)?.get(0, 0)?;
                for (c_expr, body) in cases {
                    let cv = self.eval_expr(c_expr, external)?.get(0, 0)?;
                    if (target_v - cv).abs() < 1e-12 {
                        return self.exec_block(body, external);
                    }
                }
                self.exec_block(otherwise, external)
            }
            Stmt::For { var, range, body } => {
                let r_tensor = self.eval_expr(range, external)?;
                let mut last = None;
                for c in 0..r_tensor.cols() {
                    for r in 0..r_tensor.rows() {
                        self.tick_fuel()?;
                        let v = r_tensor.get(r, c)?;
                        let scalar = Tensor::scalar(v, r_tensor.unit(), &self.budget)?;
                        self.vars.insert(var.clone(), scalar);
                        let (val, sig) = self.exec_block(body, external)?;
                        if val.is_some() {
                            last = val;
                        }
                        match sig {
                            FlowSignal::None | FlowSignal::Continue => {}
                            FlowSignal::Break => return Ok((last, FlowSignal::None)),
                            FlowSignal::Return => return Ok((last, FlowSignal::Return)),
                        }
                    }
                }
                Ok((last, FlowSignal::None))
            }
            Stmt::While { cond, body } => {
                let mut last = None;
                loop {
                    self.tick_fuel()?;
                    let c = self.eval_expr(cond, external)?;
                    if c.is_empty() || c.get(0, 0)? == 0.0 {
                        break;
                    }
                    let (val, sig) = self.exec_block(body, external)?;
                    if val.is_some() {
                        last = val;
                    }
                    match sig {
                        FlowSignal::None | FlowSignal::Continue => {}
                        FlowSignal::Break => break,
                        FlowSignal::Return => return Ok((last, FlowSignal::Return)),
                    }
                }
                Ok((last, FlowSignal::None))
            }
        }
    }

    fn exec_block<P: ExternalBuiltinProvider>(
        &mut self,
        stmts: &[Stmt],
        external: &mut P,
    ) -> Result<(Option<Tensor<f64>>, FlowSignal), LangError> {
        let mut last = None;
        for s in stmts {
            let (val, sig) = self.exec_stmt(s, external)?;
            if val.is_some() {
                last = val;
            }
            if sig != FlowSignal::None {
                return Ok((last, sig));
            }
        }
        Ok((last, FlowSignal::None))
    }

    fn resolve_index_list<P: ExternalBuiltinProvider>(
        &mut self,
        idx_expr: &Expr,
        dim_len: usize,
        external: &mut P,
    ) -> Result<Vec<usize>, LangError> {
        if matches!(idx_expr, Expr::ColonAll) {
            return Ok((0..dim_len).collect());
        }
        self.active_end_len = Some(dim_len);
        let t = self.eval_expr(idx_expr, external)?;
        self.active_end_len = None;
        let mut out = Vec::with_capacity(t.numel());
        for i in 0..t.numel() {
            let v1 = t.get_linear(i)? as usize;
            if v1 == 0 || v1 > dim_len {
                return Err(LangError::Tensor(TensorError::IndexOutOfBounds {
                    index: vec![v1],
                    shape: vec![dim_len],
                }));
            }
            out.push(v1 - 1);
        }
        Ok(out)
    }

    fn call_multi_output_function<P: ExternalBuiltinProvider>(
        &mut self,
        name: &str,
        args: &[Tensor<f64>],
        external: &mut P,
    ) -> Result<Vec<Tensor<f64>>, LangError> {
        if let Some(uf) = self.functions.get(name).cloned() {
            return self.invoke_user_function(&uf, args, external);
        }
        if let Some(res) = external.call_external(name, args, &self.budget)? {
            return Ok(res);
        }
        let single = self.call_l1_builtin(name, args)?;
        Ok(vec![single])
    }

    fn invoke_user_function<P: ExternalBuiltinProvider>(
        &mut self,
        uf: &UserFunction,
        args: &[Tensor<f64>],
        external: &mut P,
    ) -> Result<Vec<Tensor<f64>>, LangError> {
        if self.call_depth >= Self::MAX_CALL_DEPTH {
            return Err(LangError::RecursionLimitExceeded {
                limit: Self::MAX_CALL_DEPTH,
            });
        }
        self.call_depth = self.call_depth.saturating_add(1);
        let saved_vars = self.vars.clone();
        self.vars.clear();
        if let Ok(pi_t) = Tensor::scalar(std::f64::consts::PI, SiUnit::DIMENSIONLESS, &self.budget)
        {
            self.vars.insert("pi".to_string(), pi_t);
        }
        for (p_name, arg_val) in uf.params.iter().zip(args) {
            self.vars.insert(p_name.clone(), arg_val.clone());
        }
        let block_res = self.exec_block(&uf.body, external);
        let mut results = Vec::with_capacity(uf.outputs.len().max(1));
        let eval_err = match block_res {
            Ok(_) => {
                if uf.outputs.is_empty() {
                    if let Some(ans) = self.vars.get("ans").cloned() {
                        results.push(ans);
                    }
                    None
                } else {
                    let mut err = None;
                    for out_name in &uf.outputs {
                        match self.vars.get(out_name).cloned() {
                            Some(val) => results.push(val),
                            None => {
                                err = Some(LangError::UndefinedSymbol(out_name.clone()));
                                break;
                            }
                        }
                    }
                    err
                }
            }
            Err(e) => Some(e),
        };
        self.vars = saved_vars;
        self.call_depth = self.call_depth.saturating_sub(1);
        if let Some(e) = eval_err {
            return Err(e);
        }
        Ok(results)
    }

    pub fn eval_expr<P: ExternalBuiltinProvider>(
        &mut self,
        expr: &Expr,
        external: &mut P,
    ) -> Result<Tensor<f64>, LangError> {
        self.tick_fuel()?;
        match expr {
            Expr::Literal(v, unit) => Ok(Tensor::scalar(*v, *unit, &self.budget)?),
            Expr::EndKeyword => {
                let end_val = self.active_end_len.unwrap_or(1) as f64;
                Ok(Tensor::scalar(end_val, SiUnit::DIMENSIONLESS, &self.budget)?)
            }
            Expr::ColonAll => {
                let len = self.active_end_len.unwrap_or(1);
                let vals: Vec<f64> = (1..=len).map(|i| i as f64).collect();
                Ok(Tensor::from_column_major(
                    1,
                    len,
                    vals,
                    SiUnit::DIMENSIONLESS,
                    &self.budget,
                )?)
            }
            Expr::AnonFunc { .. } => Ok(Tensor::scalar(0.0, SiUnit::DIMENSIONLESS, &self.budget)?),
            Expr::Var(name) => self
                .vars
                .get(name)
                .cloned()
                .ok_or_else(|| LangError::UndefinedSymbol(name.clone())),
            Expr::Matrix(rows) => {
                if rows.is_empty() {
                    return Ok(Tensor::zeros(0, 0, &self.budget)?);
                }
                let mut row_tensors: Vec<Tensor<f64>> = Vec::with_capacity(rows.len());
                for r in rows {
                    let mut current_row_t: Option<Tensor<f64>> = None;
                    for item in r {
                        let t = self.eval_expr(item, external)?;
                        current_row_t = Some(match current_row_t {
                            None => t,
                            Some(prev) => prev.horzcat(&t)?,
                        });
                    }
                    if let Some(rt) = current_row_t {
                        row_tensors.push(rt);
                    }
                }
                let mut iter = row_tensors.into_iter();
                let Some(mut acc) = iter.next() else {
                    return Ok(Tensor::zeros(0, 0, &self.budget)?);
                };
                for next_r in iter {
                    acc = acc.vertcat(&next_r)?;
                }
                Ok(acc)
            }
            Expr::Range { start, step, stop } => {
                let a = self.eval_expr(start, external)?;
                let b = self.eval_expr(stop, external)?;
                let unit = a.unit().verify_same(b.unit(), ":")?;
                let start_v = a.get(0, 0)?;
                let stop_v = b.get(0, 0)?;
                let step_v = if let Some(st) = step {
                    let s_t = self.eval_expr(st, external)?;
                    s_t.unit().verify_same(unit, ":")?;
                    s_t.get(0, 0)?
                } else {
                    1.0
                };
                if step_v == 0.0 || !step_v.is_finite() {
                    return Err(LangError::BuiltinError {
                        name: "range".into(),
                        reason: "Step size cannot be zero or non-finite".into(),
                    });
                }
                let count_f = ((stop_v - start_v) / step_v).floor() + 1.0;
                if count_f <= 0.0 {
                    return Ok(Tensor::zeros(1, 0, &self.budget)?);
                }
                let n = count_f as usize;
                let _res = self.budget.try_reserve_elements(n, 8).map_err(TensorError::from)?;
                let vals: Vec<f64> = (0..n).map(|i| start_v + (i as f64) * step_v).collect();
                Ok(Tensor::from_column_major(1, n, vals, unit, &self.budget)?)
            }
            Expr::UnaryNeg(inner) => {
                let t = self.eval_expr(inner, external)?;
                Ok(t.map_unary(t.unit(), |x| -x)?)
            }
            Expr::LogicalNot(inner) => {
                let t = self.eval_expr(inner, external)?;
                Ok(t.map_unary(SiUnit::DIMENSIONLESS, |x| if x == 0.0 { 1.0 } else { 0.0 })?)
            }
            Expr::Transpose(inner) => {
                let t = self.eval_expr(inner, external)?;
                Ok(t.transpose()?)
            }
            Expr::Binary { op, lhs, rhs } => {
                let l = self.eval_expr(lhs, external)?;
                let r = self.eval_expr(rhs, external)?;
                match op {
                    BinOp::Add => Ok(l.checked_add(&r)?),
                    BinOp::Sub => Ok(l.checked_sub(&r)?),
                    BinOp::ElemMul => Ok(l.elem_mul(&r)?),
                    BinOp::ElemDiv => Ok(l.elem_div(&r)?),
                    BinOp::ElemPow => {
                        r.unit().verify_dimensionless(".^")?;
                        let exp = r.get(0, 0)?;
                        let out_unit = l.unit().checked_pow(exp as i8)?;
                        Ok(l.map_unary(out_unit, |x| x.powf(exp))?)
                    }
                    BinOp::Pow => {
                        r.unit().verify_dimensionless("^")?;
                        let exp = r.get(0, 0)?;
                        if l.is_scalar() {
                            let out_unit = l.unit().checked_pow(exp as i8)?;
                            Ok(l.map_unary(out_unit, |x| x.powf(exp))?)
                        } else {
                            if l.rows() != l.cols() {
                                return Err(LangError::Tensor(TensorError::ShapeMismatch {
                                    op: "^",
                                    lhs: l.shape().to_vec(),
                                    rhs: l.shape().to_vec(),
                                }));
                            }
                            let k = exp.round() as i64;
                            if (exp - (k as f64)).abs() > 1e-12 || k < 0 {
                                return Err(LangError::Syntax {
                                    pos: 0,
                                    message: "Matrix power requires non-negative integer exponent".into(),
                                });
                            }
                            if k == 0 {
                                return Ok(Tensor::eye(l.rows(), &self.budget)?);
                            }
                            let mut acc = l.clone();
                            for _ in 1..k {
                                if let Some(mut res) =
                                    external.call_external("matmul", &[acc.clone(), l.clone()], &self.budget)?
                                {
                                    if let Some(first) = res.pop() {
                                        acc = first;
                                        continue;
                                    }
                                }
                                acc = matmul_fallback(&acc, &l, &self.budget)?;
                            }
                            Ok(acc)
                        }
                    }
                    BinOp::MatMul => {
                        if l.is_scalar() || r.is_scalar() {
                            return Ok(l.elem_mul(&r)?);
                        }
                        if let Some(mut res) =
                            external.call_external("matmul", &[l.clone(), r.clone()], &self.budget)?
                        {
                            if let Some(first) = res.pop() {
                                return Ok(first);
                            }
                        }
                        matmul_fallback(&l, &r, &self.budget)
                    }
                    BinOp::RightDiv => {
                        if r.is_scalar() {
                            return Ok(l.elem_div(&r)?);
                        }
                        Err(LangError::BuiltinError {
                            name: "mrdivide".into(),
                            reason: "Matrix right-division requires scalar or L2 solver".into(),
                        })
                    }
                    BinOp::LeftDiv => {
                        if let Some(mut res) =
                            external.call_external("mldivide", &[l, r], &self.budget)?
                        {
                            if let Some(first) = res.pop() {
                                return Ok(first);
                            }
                        }
                        Err(LangError::BuiltinError {
                            name: "mldivide".into(),
                            reason: "No L2 solver provider attached".into(),
                        })
                    }
                    BinOp::EqEq => Ok(l.compare_mask(&r, "==", |a, b| (a - b).abs() < 1e-12)?),
                    BinOp::NotEq => Ok(l.compare_mask(&r, "~=", |a, b| (a - b).abs() >= 1e-12)?),
                    BinOp::Lt => Ok(l.compare_mask(&r, "<", |a, b| a < b)?),
                    BinOp::Le => Ok(l.compare_mask(&r, "<=", |a, b| a <= b)?),
                    BinOp::Gt => Ok(l.compare_mask(&r, ">", |a, b| a > b)?),
                    BinOp::Ge => Ok(l.compare_mask(&r, ">=", |a, b| a >= b)?),
                    BinOp::And => Ok(l.compare_mask(&r, "&&", |a, b| a != 0.0 && b != 0.0)?),
                    BinOp::Or => Ok(l.compare_mask(&r, "||", |a, b| a != 0.0 || b != 0.0)?),
                }
            }
            Expr::CallOrIndex { name, args } => {
                // 1. Check anonymous function handles `@(x) ...`
                if let Some((params, body_expr)) = self.anon_funcs.get(name).cloned() {
                    if self.call_depth >= Self::MAX_CALL_DEPTH {
                        return Err(LangError::RecursionLimitExceeded {
                            limit: Self::MAX_CALL_DEPTH,
                        });
                    }
                    self.call_depth = self.call_depth.saturating_add(1);
                    let eval_args: Vec<Tensor<f64>> = args
                        .iter()
                        .map(|a| self.eval_expr(a, external))
                        .collect::<Result<Vec<_>, _>>()?;
                    let saved = self.vars.clone();
                    for (p, val) in params.iter().zip(&eval_args) {
                        self.vars.insert(p.clone(), val.clone());
                    }
                    let res = self.eval_expr(&body_expr, external);
                    self.vars = saved;
                    self.call_depth = self.call_depth.saturating_sub(1);
                    return res;
                }

                // 2. Check if `name` is a variable in workspace (supports slices, `end`, and logical masks!)
                if let Some(var_tensor) = self.vars.get(name).cloned() {
                    if args.len() == 2 {
                        let r_idx = self.resolve_index_list(
                            args.first().ok_or_else(|| LangError::UndefinedSymbol(name.clone()))?,
                            var_tensor.rows(),
                            external,
                        )?;
                        let c_idx = self.resolve_index_list(
                            args.get(1).ok_or_else(|| LangError::UndefinedSymbol(name.clone()))?,
                            var_tensor.cols(),
                            external,
                        )?;
                        return Ok(var_tensor.slice_by_indices(&r_idx, &c_idx)?);
                    } else if args.len() == 1 {
                        let first_arg =
                            args.first().ok_or_else(|| LangError::UndefinedSymbol(name.clone()))?;
                        if matches!(first_arg, Expr::ColonAll) {
                            return Ok(var_tensor.reshape(var_tensor.numel(), 1)?);
                        }
                        self.active_end_len = Some(var_tensor.numel());
                        let idx_t = self.eval_expr(first_arg, external)?;
                        self.active_end_len = None;
                        if idx_t.shape() == var_tensor.shape()
                            && idx_t
                                .to_column_major_vec()?
                                .iter()
                                .all(|&v| v == 0.0 || v == 1.0)
                        {
                            return Ok(var_tensor.filter_by_mask(&idx_t)?);
                        }
                        let mut out = Vec::with_capacity(idx_t.numel());
                        for i in 0..idx_t.numel() {
                            let idx1 = idx_t.get_linear(i)? as usize;
                            if idx1 == 0 {
                                return Err(LangError::Tensor(TensorError::IndexOutOfBounds {
                                    index: vec![0],
                                    shape: var_tensor.shape().to_vec(),
                                }));
                            }
                            out.push(var_tensor.get_linear(idx1 - 1)?);
                        }
                        return Ok(Tensor::from_column_major(
                            idx_t.rows(),
                            idx_t.cols(),
                            out,
                            var_tensor.unit(),
                            &self.budget,
                        )?);
                    }
                }

                let eval_args: Vec<Tensor<f64>> = args
                    .iter()
                    .map(|a| self.eval_expr(a, external))
                    .collect::<Result<Vec<_>, _>>()?;

                // 3. Check user-defined functions
                if let Some(uf) = self.functions.get(name).cloned() {
                    let mut outs = self.invoke_user_function(&uf, &eval_args, external)?;
                    return outs.drain(..).next().ok_or_else(|| LangError::BuiltinError {
                        name: name.clone(),
                        reason: "Function returned no output".into(),
                    });
                }

                // 4. Check external L2/L3/L4 provider
                if let Some(mut res) = external.call_external(name, &eval_args, &self.budget)? {
                    if let Some(first) = res.drain(..).next() {
                        return Ok(first);
                    }
                }

                // 5. Built-in L1 functions
                self.call_l1_builtin(name, &eval_args)
            }
        }
    }

    fn call_l1_builtin(
        &self,
        name: &str,
        args: &[Tensor<f64>],
    ) -> Result<Tensor<f64>, LangError> {
        match name {
            "zeros" | "ones" => {
                let fill = if name == "ones" { 1.0 } else { 0.0 };
                let (r, c) = match args.len() {
                    1 => {
                        let n = args.first().map_or(Ok(0.0), |t| t.get(0, 0))? as usize;
                        (n, n)
                    }
                    2 => {
                        let r = args.first().map_or(Ok(0.0), |t| t.get(0, 0))? as usize;
                        let c = args.get(1).map_or(Ok(0.0), |t| t.get(0, 0))? as usize;
                        (r, c)
                    }
                    _ => {
                        return Err(LangError::BuiltinError {
                            name: name.into(),
                            reason: "Expected 1 or 2 dimension arguments".into(),
                        })
                    }
                };
                Ok(Tensor::full_with_budget(
                    r,
                    c,
                    fill,
                    SiUnit::DIMENSIONLESS,
                    &self.budget,
                )?)
            }
            "eye" => {
                let n = args.first().map_or(Ok(0.0), |t| t.get(0, 0))? as usize;
                Ok(Tensor::eye(n, &self.budget)?)
            }
            "linspace" => {
                if args.len() < 2 {
                    return Err(LangError::BuiltinError {
                        name: "linspace".into(),
                        reason: "Expected (start, stop, [n])".into(),
                    });
                }
                let a = args
                    .first()
                    .ok_or_else(|| LangError::UndefinedSymbol("start".into()))?;
                let b = args
                    .get(1)
                    .ok_or_else(|| LangError::UndefinedSymbol("stop".into()))?;
                let unit = a.unit().verify_same(b.unit(), "linspace")?;
                let n = if let Some(n_arg) = args.get(2) {
                    n_arg.get(0, 0)? as usize
                } else {
                    100
                };
                Ok(Tensor::linspace(
                    a.get(0, 0)?,
                    b.get(0, 0)?,
                    n,
                    unit,
                    &self.budget,
                )?)
            }
            "reshape" => {
                if args.len() != 3 {
                    return Err(LangError::BuiltinError {
                        name: "reshape".into(),
                        reason: "Expected reshape(A, rows, cols)".into(),
                    });
                }
                let a = args.first().ok_or_else(|| LangError::UndefinedSymbol("A".into()))?;
                let r = args.get(1).map_or(Ok(0.0), |t| t.get(0, 0))? as usize;
                let c = args.get(2).map_or(Ok(0.0), |t| t.get(0, 0))? as usize;
                Ok(a.reshape(r, c)?)
            }
            "sin" | "cos" | "tan" | "exp" | "ln" | "log" | "log10" => {
                let x = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: name.into(),
                    reason: "Missing argument".into(),
                })?;
                if !x.unit().is_dimensionless() && x.unit() != SiUnit::RADIAN {
                    return Err(LangError::Unit(UnitError::RequiresDimensionless {
                        func: "trig/transcendental",
                        actual: x.unit(),
                    }));
                }
                let f: fn(f64) -> f64 = match name {
                    "sin" => f64::sin,
                    "cos" => f64::cos,
                    "tan" => f64::tan,
                    "exp" => f64::exp,
                    "log10" => f64::log10,
                    _ => f64::ln,
                };
                Ok(x.map_unary(SiUnit::DIMENSIONLESS, f)?)
            }
            "abs" => {
                let x = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "abs".into(),
                    reason: "Missing argument".into(),
                })?;
                Ok(x.map_unary(x.unit(), f64::abs)?)
            }
            "sqrt" => {
                let x = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "sqrt".into(),
                    reason: "Missing argument".into(),
                })?;
                let out_unit = x.unit().checked_sqrt()?;
                Ok(x.map_unary(out_unit, f64::sqrt)?)
            }
            "sum" => {
                let x = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "sum".into(),
                    reason: "Missing argument".into(),
                })?;
                Ok(Tensor::scalar(x.sum_all()?, x.unit(), &self.budget)?)
            }
            "mean" => {
                let x = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "mean".into(),
                    reason: "Missing argument".into(),
                })?;
                Ok(Tensor::scalar(x.mean_all()?, x.unit(), &self.budget)?)
            }
            "min" => {
                let x = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "min".into(),
                    reason: "Missing argument".into(),
                })?;
                Ok(Tensor::scalar(x.min_all()?, x.unit(), &self.budget)?)
            }
            "max" => {
                let x = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "max".into(),
                    reason: "Missing argument".into(),
                })?;
                Ok(Tensor::scalar(x.max_all()?, x.unit(), &self.budget)?)
            }
            "std" => {
                let x = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "std".into(),
                    reason: "Missing argument".into(),
                })?;
                Ok(Tensor::scalar(x.std_all()?, x.unit(), &self.budget)?)
            }
            "norm" => {
                let x = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "norm".into(),
                    reason: "Missing argument".into(),
                })?;
                Ok(Tensor::scalar(x.norm_frobenius()?, x.unit(), &self.budget)?)
            }
            "size" => {
                let x = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: "size".into(),
                    reason: "Missing argument".into(),
                })?;
                Ok(Tensor::from_column_major(
                    1,
                    2,
                    vec![x.rows() as f64, x.cols() as f64],
                    SiUnit::DIMENSIONLESS,
                    &self.budget,
                )?)
            }
            "length" | "numel" => {
                let x = args.first().ok_or_else(|| LangError::BuiltinError {
                    name: name.into(),
                    reason: "Missing argument".into(),
                })?;
                let v = if name == "length" {
                    x.rows().max(x.cols())
                } else {
                    x.numel()
                };
                Ok(Tensor::scalar(v as f64, SiUnit::DIMENSIONLESS, &self.budget)?)
            }
            other => Err(LangError::UndefinedSymbol(other.to_string())),
        }
    }
}

fn matmul_fallback(
    a: &Tensor<f64>,
    b: &Tensor<f64>,
    budget: &Arc<MemoryBudget>,
) -> Result<Tensor<f64>, LangError> {
    if a.cols() != b.rows() {
        return Err(LangError::Tensor(TensorError::ShapeMismatch {
            op: "*",
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        }));
    }
    let out_unit = a.unit().checked_mul(b.unit())?;
    let m = a.rows();
    let k = a.cols();
    let n = b.cols();
    let mut out = Tensor::full_with_budget(m, n, 0.0, out_unit, budget)?;
    for j in 0..n {
        for i in 0..m {
            let mut acc = 0.0;
            for p in 0..k {
                acc += a.get(i, p)? * b.get(p, j)?;
            }
            out.set(i, j, acc)?;
        }
    }
    Ok(out)
}
