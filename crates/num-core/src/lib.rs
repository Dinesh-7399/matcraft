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

pub mod budget;
pub mod complex;
pub mod rational;
pub mod stats;
pub mod unit;

pub use budget::{AllocError, AllocReservation, MemoryBudget};
pub use complex::Complex64;
pub use rational::Rational128;
pub use stats::{mean, variance, std_dev, median, percentile, mode_f64, covariance, pearson_r, DescriptiveStats};
pub use unit::{SiUnit, UnitError};
