use linalg_cpu::{fft_1d, ifft_1d};
use num_core::Complex64;
use std::f64::consts::PI;

/// Analytic Signal & Instantaneous Envelope via FFT Hilbert Transform (`hilbert(x)`).
#[must_use]
pub fn hilbert_analytic(x: &[f64]) -> Vec<Complex64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let c_in: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    let mut spec = fft_1d(&c_in);

    // Zero negative frequencies and double positive frequencies
    let half = n / 2;
    for (k, val) in spec.iter_mut().enumerate() {
        if k == 0 || (n.is_multiple_of(2) && k == half) {
            // Keep DC and Nyquist unchanged
        } else if k < half {
            *val = *val * Complex64::new(2.0, 0.0);
        } else {
            *val = Complex64::ZERO;
        }
    }
    ifft_1d(&spec)
}

/// Cross-Correlation sequence $R_{xy}[m] = \sum_n x[n + m] y[n]$ (`xcorr(x, y)`).
#[must_use]
pub fn xcorr(x: &[f64], y: &[f64]) -> Vec<f64> {
    let y_ref = if y.is_empty() { x } else { y };
    let n = x.len().max(y_ref.len());
    if n == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(2 * n - 1);
    for lag in -((n as isize) - 1)..=(n as isize - 1) {
        let mut sum = 0.0;
        for idx in 0..n {
            let shifted = (idx as isize) + lag;
            if shifted >= 0 && (shifted as usize) < x.len() && idx < y_ref.len() {
                sum += x.get(shifted as usize).copied().unwrap_or(0.0)
                    * y_ref.get(idx).copied().unwrap_or(0.0);
            }
        }
        out.push(sum);
    }
    out
}

/// Short-Time Fourier Transform (`spectrogram(x, window_len, hop_size)`).
/// Returns a 2D time-frequency magnitude matrix (each inner `Vec<f64>` is one time frame's spectrum).
#[must_use]
pub fn spectrogram_stft(x: &[f64], window_len: usize, hop_size: usize) -> Vec<Vec<f64>> {
    let w_len = window_len.max(4).next_power_of_two();
    let hop = hop_size.max(1);
    let mut frames = Vec::new();
    let mut start = 0usize;

    while start + w_len <= x.len() {
        let mut frame = Vec::with_capacity(w_len);
        for i in 0..w_len {
            let sample = x.get(start + i).copied().unwrap_or(0.0);
            let hann = 0.5 * (1.0 - (2.0 * PI * (i as f64) / ((w_len - 1) as f64)).cos());
            frame.push(Complex64::new(sample * hann, 0.0));
        }
        let spec = fft_1d(&frame);
        let mag: Vec<f64> = spec.iter().take(w_len / 2 + 1).map(|c| c.abs()).collect();
        frames.push(mag);
        start += hop;
    }
    frames
}

/// 1D Natural Cubic Spline Interpolation (`interp1(x, y, xq, 'spline')`).
#[must_use]
pub fn interp1_cubic_spline(x: &[f64], y: &[f64], x_query: &[f64]) -> Vec<f64> {
    let n = x.len().min(y.len());
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![y.first().copied().unwrap_or(0.0); x_query.len()];
    }

    // Solve tridiagonal system for natural spline second derivatives M_i
    let mut h = vec![0.0; n - 1];
    for i in 0..(n - 1) {
        if let Some(slot) = h.get_mut(i) {
            *slot = (x.get(i + 1).copied().unwrap_or(1.0) - x.get(i).copied().unwrap_or(0.0))
                .max(1e-12);
        }
    }

    let mut alpha = vec![0.0; n];
    for i in 1..(n - 1) {
        let hi = h.get(i).copied().unwrap_or(1.0);
        let hi_prev = h.get(i - 1).copied().unwrap_or(1.0);
        let yi_next = y.get(i + 1).copied().unwrap_or(0.0);
        let yi = y.get(i).copied().unwrap_or(0.0);
        let yi_prev = y.get(i - 1).copied().unwrap_or(0.0);
        if let Some(slot) = alpha.get_mut(i) {
            *slot = (3.0 / hi) * (yi_next - yi) - (3.0 / hi_prev) * (yi - yi_prev);
        }
    }

    let mut l = vec![1.0; n];
    let mut mu = vec![0.0; n];
    let mut z = vec![0.0; n];

    for i in 1..(n - 1) {
        let hi = h.get(i).copied().unwrap_or(1.0);
        let hi_prev = h.get(i - 1).copied().unwrap_or(1.0);
        let li = 2.0 * (x.get(i + 1).copied().unwrap_or(1.0) - x.get(i - 1).copied().unwrap_or(0.0))
            - hi_prev * mu.get(i - 1).copied().unwrap_or(0.0);
        if let Some(slot) = l.get_mut(i) {
            *slot = li;
        }
        if let Some(slot) = mu.get_mut(i) {
            *slot = hi / li;
        }
        let z_prev = z.get(i - 1).copied().unwrap_or(0.0);
        let alpha_i = alpha.get(i).copied().unwrap_or(0.0);
        if let Some(slot) = z.get_mut(i) {
            *slot = (alpha_i - hi_prev * z_prev) / li;
        }
    }

    let mut c = vec![0.0; n];
    let mut b = vec![0.0; n - 1];
    let mut d = vec![0.0; n - 1];

    for j in (0..(n - 1)).rev() {
        let c_next = c.get(j + 1).copied().unwrap_or(0.0);
        let cj = z.get(j).copied().unwrap_or(0.0)
            - mu.get(j).copied().unwrap_or(0.0) * c_next;
        if let Some(slot) = c.get_mut(j) {
            *slot = cj;
        }
        let hj = h.get(j).copied().unwrap_or(1.0);
        let cj_next = c.get(j + 1).copied().unwrap_or(0.0);
        let yj_next = y.get(j + 1).copied().unwrap_or(0.0);
        let yj = y.get(j).copied().unwrap_or(0.0);
        if let Some(b_slot) = b.get_mut(j) {
            *b_slot = (yj_next - yj) / hj - hj * (cj_next + 2.0 * cj) / 3.0;
        }
        if let Some(d_slot) = d.get_mut(j) {
            *d_slot = (cj_next - cj) / (3.0 * hj);
        }
    }

    let mut out = Vec::with_capacity(x_query.len());
    for &xq in x_query {
        let mut seg = 0usize;
        for i in 0..(n - 1) {
            if xq >= x.get(i).copied().unwrap_or(0.0) {
                seg = i;
            }
        }
        let dx = xq - x.get(seg).copied().unwrap_or(0.0);
        let val = y.get(seg).copied().unwrap_or(0.0)
            + b.get(seg).copied().unwrap_or(0.0) * dx
            + c.get(seg).copied().unwrap_or(0.0) * dx * dx
            + d.get(seg).copied().unwrap_or(0.0) * dx * dx * dx;
        out.push(val);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hilbert_envelope_and_cubic_spline() {
        let n = 64;
        let sig: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 4.0 * (i as f64) / (n as f64)).cos())
            .collect();
        let analytic = hilbert_analytic(&sig);
        // Magnitude of analytic signal of cos(wt) is identically 1.0!
        let env_mid = analytic.get(n / 2).copied().unwrap_or_default().abs();
        assert!((env_mid - 1.0).abs() < 1e-9);

        // Cubic spline reproduces quadratic y = x^2 with high accuracy
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let y_interp = interp1_cubic_spline(&xs, &ys, &[1.5, 2.5]);
        assert!((y_interp.first().copied().unwrap_or(0.0) - 2.25).abs() < 5e-2);
        assert!((y_interp.get(1).copied().unwrap_or(0.0) - 6.25).abs() < 5e-2);
    }
}
