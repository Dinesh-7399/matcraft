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

//! Multi-rate Signal Processing and Rational Resampling.
//! Implements polyphase rational sample rate conversion (`resample(x, p, q)`),
//! anti-aliased decimation (`decimate`), and upsampling/downsampling primitives.

use std::f64::consts::PI;

/// Computes the greatest common divisor of two integers.
#[must_use]
pub fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

/// Downsamples a discrete sequence by integer factor `m` (`downsample(x, m, phase)`).
#[must_use]
pub fn downsample(x: &[f64], factor: usize, phase: usize) -> Vec<f64> {
    if factor == 0 || x.is_empty() {
        return Vec::new();
    }
    let p = phase % factor;
    let mut out = Vec::with_capacity(x.len().saturating_sub(p).div_ceil(factor));
    let mut idx = p;
    while idx < x.len() {
        if let Some(&val) = x.get(idx) {
            out.push(val);
        }
        idx = idx.saturating_add(factor);
    }
    out
}

/// Upsamples a discrete sequence by integer factor `l` by inserting zeros (`upsample(x, l, phase)`).
#[must_use]
pub fn upsample(x: &[f64], factor: usize, phase: usize) -> Vec<f64> {
    if factor == 0 || x.is_empty() {
        return Vec::new();
    }
    let p = phase % factor;
    let mut out = vec![0.0; x.len().saturating_mul(factor)];
    for (i, &val) in x.iter().enumerate() {
        let out_idx = i.saturating_mul(factor).saturating_add(p);
        if let Some(slot) = out.get_mut(out_idx) {
            *slot = val;
        }
    }
    out
}

/// Designs a low-pass anti-aliasing / reconstruction FIR filter using a Hann-windowed sinc kernel.
#[must_use]
pub fn design_resample_filter(p: usize, q: usize, half_len: usize) -> Vec<f64> {
    let max_rate = p.max(q) as f64;
    let cutoff = 1.0 / max_rate;
    let n_taps = 2 * half_len * p + 1;
    let mut h = Vec::with_capacity(n_taps);
    let center = (half_len * p) as f64;

    let mut sum = 0.0;
    for i in 0..n_taps {
        let t = (i as f64) - center;
        let sinc = if t.abs() < 1e-12 {
            1.0
        } else {
            (PI * cutoff * t).sin() / (PI * cutoff * t)
        };
        // Hann window over the entire tap length
        let w = 0.5 * (1.0 - (2.0 * PI * (i as f64) / ((n_taps - 1) as f64)).cos());
        let val = sinc * w * (p as f64);
        h.push(val);
        sum += val;
    }

    // Normalize so DC gain is exactly p (compensates for upsampling zero-insertion)
    if sum.abs() > 1e-12 {
        let scale = (p as f64) / sum;
        for val in &mut h {
            *val *= scale;
        }
    }

    h
}

/// Polyphase rational resampling of a signal by rational factor $P / Q$ (`resample(x, p, q)`).
///
/// Changes the sample rate from $F_{s1}$ to $F_{s2} = F_{s1} \cdot P / Q$.
/// Automatically inserts an anti-aliasing / anti-imaging polyphase FIR filter to prevent
/// Nyquist spectral folding.
#[must_use]
pub fn resample(x: &[f64], p: usize, q: usize) -> Vec<f64> {
    if x.is_empty() || p == 0 || q == 0 {
        return Vec::new();
    }

    // Simplify rational fraction
    let d = gcd(p, q);
    let p_red = p / d;
    let q_red = q / d;

    if p_red == 1 && q_red == 1 {
        return x.to_vec();
    }

    // Half length of filter in input samples
    let half_len = 10usize;
    let filter = design_resample_filter(p_red, q_red, half_len);

    // Theoretical output length
    let out_len = ((x.len() as u64) * (p_red as u64) / (q_red as u64)) as usize;
    let mut out = Vec::with_capacity(out_len);

    // Apply polyphase filtering without materializing the huge upsampled zero array
    for out_idx in 0..out_len {
        let src_fractional = (out_idx as f64) * (q_red as f64) / (p_red as f64);
        let src_center = src_fractional.floor() as isize;
        let poly_phase = (out_idx * q_red) % p_red;

        let mut acc = 0.0;
        let p_phase = poly_phase;

        for k in -(half_len as isize)..=(half_len as isize) {
            let input_idx = src_center + k;
            if input_idx >= 0 && (input_idx as usize) < x.len() {
                // Corresponding tap in the prototype filter
                let tap_idx = ((k + (half_len as isize)) as usize) * p_red + p_phase;
                if let (Some(&x_val), Some(&h_val)) = (x.get(input_idx as usize), filter.get(tap_idx)) {
                    acc += x_val * h_val;
                }
            }
        }
        out.push(acc);
    }

    out
}

/// Decimates a signal by integer factor `r` (`decimate(x, r)`).
/// Applies a moving-average / FIR low-pass anti-aliasing filter before downsampling.
#[must_use]
pub fn decimate(x: &[f64], r: usize) -> Vec<f64> {
    if r <= 1 || x.is_empty() {
        return x.to_vec();
    }

    // Design simple moving average anti-aliasing FIR of length 2*r
    let filter_len = 2 * r;
    let mut filtered = Vec::with_capacity(x.len());
    let mut acc = 0.0;

    for (i, &val) in x.iter().enumerate() {
        acc += val;
        if i >= filter_len {
            if let Some(&old) = x.get(i - filter_len) {
                acc -= old;
            }
        }
        let count = (i + 1).min(filter_len) as f64;
        filtered.push(acc / count);
    }

    downsample(&filtered, r, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd_and_downsample_upsample() {
        assert_eq!(gcd(12, 18), 6);
        assert_eq!(gcd(100, 25), 25);
        assert_eq!(gcd(7, 13), 1);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let down = downsample(&data, 2, 0);
        assert_eq!(down, vec![1.0, 3.0, 5.0]);

        let up = upsample(&down, 2, 0);
        assert_eq!(up, vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0]);
    }

    #[test]
    fn test_rational_resample_dc_and_sine() {
        // 1. Resample a constant DC signal: DC level must be preserved
        let dc_in = vec![5.0; 50];
        let dc_out = resample(&dc_in, 3, 2); // Upsample 1.5x
        assert_eq!(dc_out.len(), 75);
        // Middle samples should be very close to 5.0 (excluding edge transient)
        for &v in dc_out.iter().skip(10).take(50) {
            assert!((v - 5.0).abs() < 0.25, "DC preservation failed: got {v}");
        }

        // 2. Resample a pure tone without aliasing
        let fs1 = 1000.0;
        let f_tone = 50.0;
        let mut sine_in = Vec::with_capacity(200);
        for i in 0..200 {
            let t = (i as f64) / fs1;
            sine_in.push((2.0 * PI * f_tone * t).sin());
        }

        // Resample from 1000 Hz to 2000 Hz (p=2, q=1)
        let sine_2k = resample(&sine_in, 2, 1);
        assert_eq!(sine_2k.len(), 400);

        // Check phase alignment at t=0.05s (index 50 in 1k, index 100 in 2k)
        let val_1k = sine_in.get(50).copied().unwrap_or(0.0);
        let val_2k = sine_2k.get(100).copied().unwrap_or(0.0);
        assert!((val_1k - val_2k).abs() < 0.15, "Resampled tone mismatch: {val_1k} vs {val_2k}");
    }

    #[test]
    fn test_decimate() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dec = decimate(&x, 2);
        assert_eq!(dec.len(), 4);
    }
}
