use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum UnitError {
    #[error("Dimensional mismatch: cannot perform `{op}` on `{lhs}` and `{rhs}`")]
    IncompatibleUnits {
        op: &'static str,
        lhs: SiUnit,
        rhs: SiUnit,
    },
    #[error("Unit exponent overflow while computing unit operation")]
    ExponentOverflow,
    #[error("Transcendental function `{func}` requires a dimensionless argument, but received `{actual}`")]
    RequiresDimensionless { func: &'static str, actual: SiUnit },
    #[error("Unknown physical unit symbol '{0}'")]
    UnknownUnitSymbol(String),
}

/// Zero-cost 8-dimensional SI unit vector:
/// `[m, kg, s, A, K, mol, cd, rad]`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct SiUnit {
    pub exponents: [i8; 8],
}

impl SiUnit {
    pub const DIMENSIONLESS: Self = Self { exponents: [0; 8] };
    pub const METER: Self = Self {
        exponents: [1, 0, 0, 0, 0, 0, 0, 0],
    };
    pub const KILOGRAM: Self = Self {
        exponents: [0, 1, 0, 0, 0, 0, 0, 0],
    };
    pub const SECOND: Self = Self {
        exponents: [0, 0, 1, 0, 0, 0, 0, 0],
    };
    pub const AMPERE: Self = Self {
        exponents: [0, 0, 0, 1, 0, 0, 0, 0],
    };
    pub const KELVIN: Self = Self {
        exponents: [0, 0, 0, 0, 1, 0, 0, 0],
    };
    pub const MOLE: Self = Self {
        exponents: [0, 0, 0, 0, 0, 1, 0, 0],
    };
    pub const CANDELA: Self = Self {
        exponents: [0, 0, 0, 0, 0, 0, 1, 0],
    };
    pub const RADIAN: Self = Self {
        exponents: [0, 0, 0, 0, 0, 0, 0, 1],
    };

    // Derived SI units
    pub const HERTZ: Self = Self {
        exponents: [0, 0, -1, 0, 0, 0, 0, 0],
    };
    pub const NEWTON: Self = Self {
        exponents: [1, 1, -2, 0, 0, 0, 0, 0],
    };
    pub const PASCAL: Self = Self {
        exponents: [-1, 1, -2, 0, 0, 0, 0, 0],
    };
    pub const JOULE: Self = Self {
        exponents: [2, 1, -2, 0, 0, 0, 0, 0],
    };
    pub const WATT: Self = Self {
        exponents: [2, 1, -3, 0, 0, 0, 0, 0],
    };
    pub const VOLT: Self = Self {
        exponents: [2, 1, -3, -1, 0, 0, 0, 0],
    };
    pub const OHM: Self = Self {
        exponents: [2, 1, -3, -2, 0, 0, 0, 0],
    };
    pub const FARAD: Self = Self {
        exponents: [-2, -1, 4, 2, 0, 0, 0, 0],
    };
    pub const HENRY: Self = Self {
        exponents: [2, 1, -2, -2, 0, 0, 0, 0],
    };

    #[must_use]
    pub const fn is_dimensionless(self) -> bool {
        let [a, b, c, d, e, f, g, h] = self.exponents;
        a == 0 && b == 0 && c == 0 && d == 0 && e == 0 && f == 0 && g == 0 && h == 0
    }

    pub fn verify_same(self, rhs: Self, op: &'static str) -> Result<Self, UnitError> {
        if self == rhs {
            Ok(self)
        } else {
            Err(UnitError::IncompatibleUnits { op, lhs: self, rhs })
        }
    }

    pub fn verify_dimensionless(self, func: &'static str) -> Result<Self, UnitError> {
        if self.is_dimensionless() {
            Ok(Self::DIMENSIONLESS)
        } else {
            Err(UnitError::RequiresDimensionless { func, actual: self })
        }
    }

    pub fn checked_mul(self, rhs: Self) -> Result<Self, UnitError> {
        let mut out = [0i8; 8];
        for ((dst, a), b) in out.iter_mut().zip(self.exponents).zip(rhs.exponents) {
            *dst = a.checked_add(b).ok_or(UnitError::ExponentOverflow)?;
        }
        Ok(Self { exponents: out })
    }

    pub fn checked_div(self, rhs: Self) -> Result<Self, UnitError> {
        let mut out = [0i8; 8];
        for ((dst, a), b) in out.iter_mut().zip(self.exponents).zip(rhs.exponents) {
            *dst = a.checked_sub(b).ok_or(UnitError::ExponentOverflow)?;
        }
        Ok(Self { exponents: out })
    }

    pub fn checked_pow(self, exp: i8) -> Result<Self, UnitError> {
        let mut out = [0i8; 8];
        for (dst, a) in out.iter_mut().zip(self.exponents) {
            *dst = a.checked_mul(exp).ok_or(UnitError::ExponentOverflow)?;
        }
        Ok(Self { exponents: out })
    }

    pub fn checked_sqrt(self) -> Result<Self, UnitError> {
        let mut out = [0i8; 8];
        for (dst, a) in out.iter_mut().zip(self.exponents) {
            if a % 2 != 0 {
                return Err(UnitError::ExponentOverflow);
            }
            *dst = a / 2;
        }
        Ok(Self { exponents: out })
    }

    /// Parses compound unit expressions like `"m/s^2"`, `"N*m"`, `"kg*m/s^2"`, `"Hz"`, `"V"`.
    pub fn parse(input: &str) -> Result<Self, UnitError> {
        let trimmed = input.trim();
        if trimmed.is_empty() || trimmed == "1" {
            return Ok(Self::DIMENSIONLESS);
        }
        let mut parts = trimmed.split('/');
        let numerator_str = parts.next().unwrap_or("");
        let mut acc = Self::parse_product(numerator_str)?;
        for denom_str in parts {
            let denom = Self::parse_product(denom_str)?;
            acc = acc.checked_div(denom)?;
        }
        Ok(acc)
    }

    fn parse_product(prod: &str) -> Result<Self, UnitError> {
        let mut acc = Self::DIMENSIONLESS;
        for term in prod.split('*') {
            let term = term.trim();
            if term.is_empty() || term == "1" {
                continue;
            }
            let (sym, exp) = if let Some((base, pow_str)) = term.split_once('^') {
                let p: i8 = pow_str
                    .trim()
                    .parse()
                    .map_err(|_| UnitError::UnknownUnitSymbol(term.to_string()))?;
                (base.trim(), p)
            } else {
                (term, 1)
            };
            let base_unit = Self::from_symbol(sym)?;
            acc = acc.checked_mul(base_unit.checked_pow(exp)?)?;
        }
        Ok(acc)
    }

    pub fn from_symbol(sym: &str) -> Result<Self, UnitError> {
        match sym {
            "m" => Ok(Self::METER),
            "kg" => Ok(Self::KILOGRAM),
            "s" => Ok(Self::SECOND),
            "A" => Ok(Self::AMPERE),
            "K" => Ok(Self::KELVIN),
            "mol" => Ok(Self::MOLE),
            "cd" => Ok(Self::CANDELA),
            "rad" => Ok(Self::RADIAN),
            "Hz" => Ok(Self::HERTZ),
            "N" => Ok(Self::NEWTON),
            "Pa" => Ok(Self::PASCAL),
            "J" => Ok(Self::JOULE),
            "W" => Ok(Self::WATT),
            "V" => Ok(Self::VOLT),
            "Ohm" => Ok(Self::OHM),
            "F" => Ok(Self::FARAD),
            "H" => Ok(Self::HENRY),
            other => Err(UnitError::UnknownUnitSymbol(other.to_string())),
        }
    }
}

impl fmt::Display for SiUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dimensionless() {
            return write!(f, "1");
        }
        if *self == Self::HERTZ {
            return write!(f, "Hz");
        }
        if *self == Self::NEWTON {
            return write!(f, "N");
        }
        if *self == Self::PASCAL {
            return write!(f, "Pa");
        }
        if *self == Self::JOULE {
            return write!(f, "J");
        }
        if *self == Self::WATT {
            return write!(f, "W");
        }
        if *self == Self::VOLT {
            return write!(f, "V");
        }
        if *self == Self::OHM {
            return write!(f, "Ohm");
        }
        let names = ["m", "kg", "s", "A", "K", "mol", "cd", "rad"];
        let mut num = Vec::new();
        let mut den = Vec::new();
        for (&exp, &name) in self.exponents.iter().zip(&names) {
            if exp > 0 {
                if exp == 1 {
                    num.push(name.to_string());
                } else {
                    num.push(format!("{name}^{exp}"));
                }
            } else if exp < 0 {
                let pos = exp.saturating_neg();
                if pos == 1 {
                    den.push(name.to_string());
                } else {
                    den.push(format!("{name}^{pos}"));
                }
            }
        }
        let num_str = if num.is_empty() {
            "1".to_string()
        } else {
            num.join("*")
        };
        if den.is_empty() {
            write!(f, "{num_str}")
        } else {
            write!(f, "{num_str}/{}", den.join("*"))
        }
    }
}
