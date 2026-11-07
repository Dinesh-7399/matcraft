use serde::{Deserialize, Serialize};
use std::fmt;

/// Exact 128-bit Rational number (`num / den` with `den > 0` and `gcd(|num|, den) == 1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rational128 {
    num: i128,
    den: i128,
}

impl Rational128 {
    pub const ZERO: Self = Self { num: 0, den: 1 };
    pub const ONE: Self = Self { num: 1, den: 1 };

    pub fn new(num: i128, den: i128) -> Option<Self> {
        if den == 0 || num == i128::MIN || den == i128::MIN {
            return None;
        }
        let sign = if den < 0 { -1 } else { 1 };
        let n = num.checked_mul(sign)?;
        let d = den.checked_abs()?;
        let g = gcd_i128(n.checked_abs()?, d);
        Some(Self {
            num: n / g,
            den: d / g,
        })
    }

    #[must_use]
    pub const fn numer(self) -> i128 {
        self.num
    }

    #[must_use]
    pub const fn denom(self) -> i128 {
        self.den
    }

    #[must_use]
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        let lhs_scaled = self.num.checked_mul(rhs.den)?;
        let rhs_scaled = rhs.num.checked_mul(self.den)?;
        let num = lhs_scaled.checked_add(rhs_scaled)?;
        let den = self.den.checked_mul(rhs.den)?;
        Self::new(num, den)
    }

    #[must_use]
    pub fn checked_mul(self, rhs: Self) -> Option<Self> {
        let num = self.num.checked_mul(rhs.num)?;
        let den = self.den.checked_mul(rhs.den)?;
        Self::new(num, den)
    }

    #[must_use]
    pub fn to_f64(self) -> f64 {
        (self.num as f64) / (self.den as f64)
    }
}

const fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    if a == 0 { 1 } else { a }
}

impl fmt::Display for Rational128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AllocError, Complex64, MemoryBudget, SiUnit};

    #[test]
    fn test_si_units_and_derived_formatting() -> Result<(), Box<dyn std::error::Error>> {
        let mass = SiUnit::parse("kg")?;
        let accel = SiUnit::parse("m/s^2")?;
        let force = mass.checked_mul(accel)?;
        assert_eq!(force, SiUnit::NEWTON);
        assert_eq!(force.to_string(), "N");

        let energy = force.checked_mul(SiUnit::METER)?;
        assert_eq!(energy, SiUnit::JOULE);
        assert_eq!(energy.to_string(), "J");

        // Adding meters to seconds must return an error, never panic
        assert!(SiUnit::METER.verify_same(SiUnit::SECOND, "+").is_err());
        Ok(())
    }

    #[test]
    fn test_panic_hunt_memory_budget_and_overflow() {
        let budget = MemoryBudget::new(1024 * 1024); // 1 MiB
        // Requesting 1,000,000 x 1,000,000 f64s (8 TB) must cleanly fail with BudgetExceeded
        let count = MemoryBudget::checked_element_count(&[1_000_000, 1_000_000]);
        assert_eq!(count, Ok(1_000_000_000_000));
        let res = budget.try_reserve_elements(1_000_000_000_000, 8);
        assert!(matches!(res, Err(AllocError::BudgetExceeded { .. })));

        // Overflowing usize::MAX dimensions
        let overflow = MemoryBudget::checked_element_count(&[usize::MAX, 2]);
        assert!(matches!(overflow, Err(AllocError::DimensionOverflow { .. })));

        // Division by zero in Complex64 and Rational128
        let c_div_zero = Complex64::ONE / Complex64::ZERO;
        assert!(c_div_zero.re.is_nan());
        assert!(Rational128::new(5, 0).is_none());
        assert!(Rational128::new(i128::MIN, -1).is_none());
    }
}
