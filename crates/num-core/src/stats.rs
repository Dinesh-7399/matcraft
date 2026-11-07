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

//! Descriptive statistics: mean, variance, std_dev, median, percentile, mode, covariance, pearson_r.

use std::collections::BTreeMap;

/// Bundle of common descriptive statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct DescriptiveStats {
    pub n: usize,
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

impl DescriptiveStats {
    /// Compute all descriptive statistics for `data`. Returns `None` if `data` is empty.
    #[must_use]
    pub fn compute(data: &[f64]) -> Option<Self> {
        let n = data.len();
        if n == 0 {
            return None;
        }
        let m = mean(data)?;
        let v = variance(data).unwrap_or(0.0);
        let sd = v.sqrt();
        let med = median(data)?;
        let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some(Self {
            n,
            mean: m,
            variance: v,
            std_dev: sd,
            median: med,
            min: min_val,
            max: max_val,
        })
    }
}

/// Arithmetic mean. Returns `None` if `data` is empty.
#[must_use]
pub fn mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    let sum: f64 = data.iter().sum();
    Some(sum / (data.len() as f64))
}

/// Sample variance (Bessel-corrected, divides by n-1). Returns `None` if fewer than 2 elements.
#[must_use]
pub fn variance(data: &[f64]) -> Option<f64> {
    let n = data.len();
    if n < 2 {
        return None;
    }
    let m = mean(data)?;
    let sum_sq: f64 = data.iter().map(|&x| (x - m) * (x - m)).sum();
    Some(sum_sq / ((n - 1) as f64))
}

/// Sample standard deviation. Returns `None` if fewer than 2 elements.
#[must_use]
pub fn std_dev(data: &[f64]) -> Option<f64> {
    variance(data).map(f64::sqrt)
}

/// Median (clones and sorts). Returns `None` if `data` is empty.
#[must_use]
pub fn median(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted.get(n / 2).copied()
    } else {
        let lo = sorted.get(n / 2 - 1).copied().unwrap_or(0.0);
        let hi = sorted.get(n / 2).copied().unwrap_or(0.0);
        Some((lo + hi) / 2.0)
    }
}

/// Percentile via linear interpolation (p in [0.0, 100.0]). Returns `None` if `data` is empty.
#[must_use]
pub fn percentile(data: &[f64], p: f64) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    let p_clamped = p.clamp(0.0, 100.0);
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return sorted.first().copied();
    }
    // Index via "exclusive" method (R type 5 / MATLAB prctile approach)
    let rank = p_clamped / 100.0 * (n as f64 - 1.0);
    let lo_idx = rank.floor() as usize;
    let hi_idx = (lo_idx + 1).min(n - 1);
    let frac = rank - rank.floor();
    let lo = sorted.get(lo_idx).copied().unwrap_or(0.0);
    let hi = sorted.get(hi_idx).copied().unwrap_or(0.0);
    Some(lo + frac * (hi - lo))
}

/// Histogram-binned mode estimate. Returns `None` if `data` is empty or `bin_width <= 0`.
#[must_use]
pub fn mode_f64(data: &[f64], bin_width: f64) -> Option<f64> {
    if data.is_empty() || bin_width <= 0.0 {
        return None;
    }
    let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
    let mut bins: BTreeMap<i64, usize> = BTreeMap::new();
    for &x in data {
        let bin_idx = ((x - min_val) / bin_width).floor() as i64;
        *bins.entry(bin_idx).or_insert(0) += 1;
    }
    let best_bin = bins.into_iter().max_by_key(|&(_, cnt)| cnt)?.0;
    Some(min_val + (best_bin as f64) * bin_width + bin_width * 0.5)
}

/// Sample covariance of two equal-length slices (Bessel-corrected). Returns `None` if fewer than 2 paired elements.
#[must_use]
pub fn covariance(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len().min(y.len());
    if n < 2 {
        return None;
    }
    let x_slice = x.get(..n)?;
    let y_slice = y.get(..n)?;
    let mx = mean(x_slice)?;
    let my = mean(y_slice)?;
    let sum: f64 = x_slice
        .iter()
        .zip(y_slice.iter())
        .map(|(&xi, &yi)| (xi - mx) * (yi - my))
        .sum();
    Some(sum / ((n - 1) as f64))
}

/// Pearson correlation coefficient. Returns `None` if fewer than 2 elements or zero variance.
#[must_use]
pub fn pearson_r(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len().min(y.len());
    if n < 2 {
        return None;
    }
    let x_slice = x.get(..n)?;
    let y_slice = y.get(..n)?;
    let cov = covariance(x_slice, y_slice)?;
    let sx = std_dev(x_slice)?;
    let sy = std_dev(y_slice)?;
    if sx.abs() < 1e-15 || sy.abs() < 1e-15 {
        return None;
    }
    Some(cov / (sx * sy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_stats() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((mean(&data).unwrap() - 5.0).abs() < 1e-12);
        assert!((variance(&data).unwrap() - 4.571_428_571_428_571).abs() < 1e-9);
        assert!((std_dev(&data).unwrap() - 2.138_089_935_618_000).abs() < 1e-9);
        assert!((median(&data).unwrap() - 4.5).abs() < 1e-12);
    }

    #[test]
    fn test_percentile_and_mode() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 0.0).unwrap() - 1.0).abs() < 1e-12);
        assert!((percentile(&data, 100.0).unwrap() - 5.0).abs() < 1e-12);
        assert!((percentile(&data, 50.0).unwrap() - 3.0).abs() < 1e-12);

        let data2 = vec![1.0, 1.0, 2.0, 3.0, 5.0];
        let m = mode_f64(&data2, 1.0);
        assert!(m.is_some());
    }

    #[test]
    fn test_covariance_and_pearson() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_r(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-12, "Perfect positive correlation expected");

        assert!(mean(&[]).is_none());
        assert!(variance(&[1.0]).is_none());
    }

    #[test]
    fn test_descriptive_stats_struct() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ds = DescriptiveStats::compute(&data).unwrap();
        assert_eq!(ds.n, 5);
        assert!((ds.mean - 3.0).abs() < 1e-12);
        assert!((ds.min - 1.0).abs() < 1e-12);
        assert!((ds.max - 5.0).abs() < 1e-12);
        assert!(DescriptiveStats::compute(&[]).is_none());
    }
}
