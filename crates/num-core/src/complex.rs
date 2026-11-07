use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// IEEE-754 double-precision complex number using Smith's overflow-safe division and hypot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    #[must_use]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[must_use]
    pub const fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[must_use]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[must_use]
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    #[must_use]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    #[must_use]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    #[must_use]
    pub fn exp(self) -> Self {
        let r = self.re.exp();
        let (s, c) = self.im.sin_cos();
        Self::new(r * c, r * s)
    }

    #[must_use]
    pub fn ln(self) -> Self {
        Self::new(self.abs().ln(), self.arg())
    }

    #[must_use]
    pub fn sqrt(self) -> Self {
        if self.re == 0.0 && self.im == 0.0 {
            return Self::ZERO;
        }
        let r = self.abs();
        let re = ((r + self.re.abs()) * 0.5).sqrt();
        let im = ((r - self.re.abs()) * 0.5).sqrt();
        if self.re >= 0.0 {
            Self::new(re, self.im.copysign(im))
        } else {
            Self::new(im, self.im.copysign(re))
        }
    }

    /// Smith's algorithm for complex division to avoid intermediate overflow/underflow.
    #[must_use]
    pub fn safe_div(self, rhs: Self) -> Self {
        if rhs.re.abs() >= rhs.im.abs() {
            if rhs.re == 0.0 {
                return Self::new(f64::NAN, f64::NAN);
            }
            let r = rhs.im / rhs.re;
            let denom = rhs.re + rhs.im * r;
            Self::new((self.re + self.im * r) / denom, (self.im - self.re * r) / denom)
        } else {
            let r = rhs.re / rhs.im;
            let denom = rhs.im + rhs.re * r;
            Self::new((self.re * r + self.im) / denom, (self.im * r - self.re) / denom)
        }
    }
}

impl Add for Complex64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Div for Complex64 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self.safe_div(rhs)
    }
}

impl Neg for Complex64 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}

impl fmt::Display for Complex64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{} + {}i", self.re, self.im)
        } else {
            write!(f, "{} - {}i", self.re, -self.im)
        }
    }
}
