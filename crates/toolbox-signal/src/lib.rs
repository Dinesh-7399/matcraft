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

pub mod multirate;
pub mod spectral;
pub mod wavelets;
pub use multirate::{decimate, downsample, resample, upsample};
pub use spectral::{hilbert_analytic, interp1_cubic_spline, spectrogram_stft, xcorr};
pub use wavelets::{dwt_1d, idwt_1d, wavedec, DB4_HI, DB4_LO};

use linalg_cpu::fft_1d;
use num_core::Complex64;
use std::f64::consts::PI;

/// Designs a normalized Hamming-windowed low-pass FIR filter (`fir1(order, cutoff_norm)`).
#[must_use]
pub fn fir1_lowpass(order: usize, cutoff_norm: f64) -> Vec<f64> {
    let n_taps = order.saturating_add(1);
    let m = (order as f64) * 0.5;
    let wc = cutoff_norm.clamp(1e-6, 0.999999) * PI;
    let mut h = Vec::with_capacity(n_taps);
    let mut sum = 0.0;

    for n in 0..n_taps {
        let x = (n as f64) - m;
        let sinc = if x.abs() < 1e-12 {
            wc / PI
        } else {
            (wc * x).sin() / (PI * x)
        };
        let hamming = if order == 0 {
            1.0
        } else {
            0.54 - 0.46 * (2.0 * PI * (n as f64) / (order as f64)).cos()
        };
        let tap = sinc * hamming;
        sum += tap;
        h.push(tap);
    }

    if sum.abs() > 1e-15 {
        for tap in &mut h {
            *tap /= sum;
        }
    }
    h
}

/// Designs a digital lowpass Butterworth IIR filter (`butter(order, cutoff_norm)`).
/// Returns `(b, a)` polynomial transfer function coefficients where `a[0] = 1.0`.
#[must_use]
pub fn butter(order: usize, cutoff_norm: f64) -> (Vec<f64>, Vec<f64>) {
    let n = order.clamp(1, 10);
    // Pre-warp cutoff frequency: W = tan(wc * pi / 2)
    let wc = cutoff_norm.clamp(1e-5, 0.9999) * (PI * 0.5);
    let w_prewarp = wc.tan();

    // Analog Butterworth poles: s_k = w_prewarp * exp(j * pi * (2k + n - 1) / (2n))
    let mut analog_poles = Vec::with_capacity(n);
    for k in 1..=n {
        let theta = PI * ((2 * k + n - 1) as f64) / (2.0 * (n as f64));
        let p = Complex64::new(w_prewarp * theta.cos(), w_prewarp * theta.sin());
        analog_poles.push(p);
    }

    // Bilinear transform: z = (1 + s/2) / (1 - s/2) with fs=2 -> z = (2 + s) / (2 - s)
    // Zeros are all at z = -1
    let mut digital_poles = Vec::with_capacity(n);
    for p in &analog_poles {
        let num = Complex64::new(2.0, 0.0) + *p;
        let den = Complex64::new(2.0, 0.0) - *p;
        digital_poles.push(num / den);
    }

    // Expand polynomial (z - p_1)(z - p_2)... for denominator
    let mut a_poly = vec![Complex64::new(1.0, 0.0)];
    for p in &digital_poles {
        let mut next = vec![Complex64::new(0.0, 0.0); a_poly.len() + 1];
        for (i, &coeff) in a_poly.iter().enumerate() {
            if let Some(slot) = next.get_mut(i) {
                *slot = *slot + coeff;
            }
            if let Some(slot) = next.get_mut(i + 1) {
                *slot = *slot - coeff * (*p);
            }
        }
        a_poly = next;
    }

    // Numerator is (1 + z^-1)^n
    let mut b_poly = vec![1.0];
    for _ in 0..n {
        let mut next = vec![0.0; b_poly.len() + 1];
        for (i, &coeff) in b_poly.iter().enumerate() {
            if let Some(slot) = next.get_mut(i) {
                *slot += coeff;
            }
            if let Some(slot) = next.get_mut(i + 1) {
                *slot += coeff;
            }
        }
        b_poly = next;
    }

    let a: Vec<f64> = a_poly.iter().map(|c| c.re).collect();
    let a0 = a.first().copied().unwrap_or(1.0);
    let a_norm: Vec<f64> = if a0.abs() > 1e-15 {
        a.iter().map(|&x| x / a0).collect()
    } else {
        a
    };

    // Normalize DC gain (at z = 1): H(1) = sum(b) / sum(a) should equal 1.0
    let sum_b: f64 = b_poly.iter().sum();
    let sum_a: f64 = a_norm.iter().sum();
    let gain = if sum_b.abs() > 1e-15 {
        sum_a / sum_b
    } else {
        1.0
    };
    let b_norm: Vec<f64> = b_poly.iter().map(|&x| x * gain).collect();

    (b_norm, a_norm)
}

/// Direct-Form II Transposed digital filter implementation (`filter(b, a, x)`).
#[must_use]
pub fn filter_df2t(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return Vec::new();
    }
    let a0 = a.first().copied().unwrap_or(1.0);
    if a0.abs() < 1e-15 {
        return x.to_vec();
    }

    let nb = b.len();
    let na = a.len();
    let order = nb.max(na);
    let mut d = vec![0.0; order];
    let mut y = Vec::with_capacity(x.len());

    for &xn in x {
        let b0 = b.first().copied().unwrap_or(0.0) / a0;
        let d0 = d.first().copied().unwrap_or(0.0);
        let yn = b0 * xn + d0;
        y.push(yn);

        for m in 0..order.saturating_sub(1) {
            let bm1 = b.get(m + 1).copied().unwrap_or(0.0) / a0;
            let am1 = a.get(m + 1).copied().unwrap_or(0.0) / a0;
            let d_next = d.get(m + 1).copied().unwrap_or(0.0);
            if let Some(slot) = d.get_mut(m) {
                *slot = d_next + bm1 * xn - am1 * yn;
            }
        }
        if let Some(last) = d.get_mut(order.saturating_sub(1)) {
            let b_last = b.get(order).copied().unwrap_or(0.0) / a0;
            let a_last = a.get(order).copied().unwrap_or(0.0) / a0;
            *last = b_last * xn - a_last * yn;
        }
    }

    y
}

/// Zero-phase digital filtering (`filtfilt(b, a, x)`).
/// Filters forward, reverses, filters backward, and reverses back.
#[must_use]
pub fn filtfilt(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return Vec::new();
    }
    // 1. Forward pass
    let y_fwd = filter_df2t(b, a, x);
    // 2. Reverse
    let mut y_rev: Vec<f64> = y_fwd.into_iter().rev().collect();
    // 3. Backward pass
    let y_back = filter_df2t(b, a, &y_rev);
    // 4. Reverse back
    y_rev = y_back.into_iter().rev().collect();
    y_rev
}

/// Welch's averaged modified periodogram power spectral density estimate (`pwelch(x, window, noverlap, nfft)`).
/// Returns `(frequencies, psd)` normalized to Nyquist ($f \in [0, 0.5]$).
#[must_use]
pub fn pwelch(
    x: &[f64],
    window_len: usize,
    noverlap: usize,
    nfft: usize,
) -> (Vec<f64>, Vec<f64>) {
    let win_len = window_len.clamp(16, 4096).min(x.len().max(16));
    let n_fft = nfft.max(win_len).next_power_of_two();
    let step = (win_len.saturating_sub(noverlap)).max(1);

    // Periodic Hann window
    let mut window = Vec::with_capacity(win_len);
    let mut s2 = 0.0;
    for n in 0..win_len {
        let w = 0.5 - 0.5 * (2.0 * PI * (n as f64) / (win_len as f64)).cos();
        s2 += w * w;
        window.push(w);
    }
    if s2 < 1e-15 {
        s2 = 1.0;
    }

    let n_bins = n_fft / 2 + 1;
    let mut psd_accum = vec![0.0; n_bins];
    let mut n_segments = 0usize;

    let mut start = 0;
    while start + win_len <= x.len() {
        let mut seg_complex = vec![Complex64::new(0.0, 0.0); n_fft];
        for i in 0..win_len {
            let xv = x.get(start + i).copied().unwrap_or(0.0);
            let wv = window.get(i).copied().unwrap_or(1.0);
            if let Some(slot) = seg_complex.get_mut(i) {
                *slot = Complex64::new(xv * wv, 0.0);
            }
        }

        let spectrum = fft_1d(&seg_complex);
        for k in 0..n_bins {
            let c = spectrum.get(k).copied().unwrap_or(Complex64::new(0.0, 0.0));
            let mag2 = c.re * c.re + c.im * c.im;
            let factor = if k == 0 || k == n_bins - 1 { 1.0 } else { 2.0 };
            if let Some(slot) = psd_accum.get_mut(k) {
                *slot += factor * mag2 / s2;
            }
        }

        n_segments += 1;
        start += step;
    }

    let divisor = (n_segments as f64).max(1.0);
    for v in &mut psd_accum {
        *v /= divisor;
    }

    let freqs: Vec<f64> = (0..n_bins)
        .map(|k| (k as f64) / (n_fft as f64))
        .collect();

    (freqs, psd_accum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_butter_filter_and_filtfilt() {
        let (b, a) = butter(2, 0.2);
        assert_eq!(b.len(), 3);
        assert_eq!(a.len(), 3);
        assert!((a.first().copied().unwrap_or(0.0) - 1.0).abs() < 1e-12);

        // Constant input should produce identical constant output at steady state
        let sig = vec![1.0; 50];
        let filtered = filter_df2t(&b, &a, &sig);
        assert_eq!(filtered.len(), 50);
        let last_val = filtered.last().copied().unwrap_or(0.0);
        assert!((last_val - 1.0).abs() < 1e-2);

        // filtfilt zero phase
        let ff = filtfilt(&b, &a, &sig);
        assert_eq!(ff.len(), 50);
        let ff_mid = ff.get(25).copied().unwrap_or(0.0);
        assert!((ff_mid - 1.0).abs() < 1e-2);
    }

    #[test]
    fn test_pwelch_and_xcorr() {
        let n = 256;
        let mut sig = Vec::with_capacity(n);
        for i in 0..n {
            let t = (i as f64) / 256.0;
            sig.push((2.0 * PI * 16.0 * t).sin());
        }
        let (freqs, psd) = pwelch(&sig, 64, 32, 128);
        assert_eq!(freqs.len(), psd.len());
        assert!(!psd.is_empty());

        let xc = xcorr(&[1.0, 2.0, 3.0], &[]);
        assert_eq!(xc.len(), 5);
        let center = xc.get(2).copied().unwrap_or(0.0);
        assert_eq!(center, 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0); // 14.0
    }
}
