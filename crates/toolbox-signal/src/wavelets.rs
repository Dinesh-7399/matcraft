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

//! Wavelet transforms: Daubechies-4 DWT/IDWT and multi-level wavedec.

/// Daubechies-4 scaling (low-pass) filter coefficients.
pub const DB4_LO: [f64; 8] = [
    -0.010_597_401_784_997_278,
     0.032_883_011_666_982_945,
     0.030_841_381_835_986_965,
    -0.187_034_811_718_881_6,
    -0.027_983_769_416_983_855,
     0.630_880_767_929_590_4,
     0.714_846_570_552_541_7,
     0.230_377_813_308_896_7,
];

/// Daubechies-4 wavelet (high-pass) filter coefficients.
/// High-pass = QMF of reversed low-pass: h[k] = (-1)^k * lo[N-1-k]
pub const DB4_HI: [f64; 8] = [
    -0.230_377_813_308_896_7,
     0.714_846_570_552_541_7,
    -0.630_880_767_929_590_4,
    -0.027_983_769_416_983_855,
     0.187_034_811_718_881_6,
     0.030_841_381_835_986_965,
    -0.032_883_011_666_982_945,
    -0.010_597_401_784_997_278,
];

/// Synthesis (reconstruction) low-pass filter = time-reversed DB4_HI
pub const DB4_LO_SYNTH: [f64; 8] = [
    -0.010_597_401_784_997_278,
    -0.032_883_011_666_982_945,
     0.030_841_381_835_986_965,
     0.187_034_811_718_881_6,
    -0.027_983_769_416_983_855,
    -0.630_880_767_929_590_4,
     0.714_846_570_552_541_7,
    -0.230_377_813_308_896_7,
];

/// Synthesis (reconstruction) high-pass filter = time-reversed DB4_LO
pub const DB4_HI_SYNTH: [f64; 8] = [
     0.230_377_813_308_896_7,
     0.714_846_570_552_541_7,
     0.630_880_767_929_590_4,
    -0.027_983_769_416_983_855,
    -0.187_034_811_718_881_6,
     0.030_841_381_835_986_965,
     0.032_883_011_666_982_945,
    -0.010_597_401_784_997_278,
];

/// Convolve signal with filter using periodic extension, then downsample by 2.
fn convolve_downsample(signal: &[f64], filter: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let flen = filter.len();
    if n == 0 || flen == 0 {
        return Vec::new();
    }
    let out_len = n.div_ceil(2);
    let out_len = out_len.max(1);
    let mut out = Vec::with_capacity(out_len);

    // We compute y[k] = sum_j filter[j] * signal[(2k - j + n*flen) % n] for k = 0..out_len
    // Using the standard "valid+periodic" convention matching PyWavelets 'periodization'
    let total = n;
    for k in 0..out_len {
        let mut acc = 0.0;
        for (j, &fj) in filter.iter().enumerate() {
            // Index into signal with periodic extension
            let raw_idx = (2 * k + j) as isize - (flen as isize - 1);
            let idx = ((raw_idx % total as isize) + total as isize) as usize % total;
            acc += fj * signal.get(idx).copied().unwrap_or(0.0);
        }
        out.push(acc);
    }
    out
}

/// Upsample by 2 (insert zeros between samples), then convolve with filter.
fn upsample_convolve(coeff: &[f64], filter: &[f64], out_len: usize) -> Vec<f64> {
    let n = coeff.len();
    let flen = filter.len();
    if n == 0 || flen == 0 || out_len == 0 {
        return vec![0.0; out_len];
    }
    // Upsampled signal (zeros between samples)
    let up_len = 2 * n;
    let mut out = vec![0.0f64; out_len];

    for k in 0..out_len {
        let mut acc = 0.0;
        for (j, &fj) in filter.iter().enumerate() {
            let raw_idx = k as isize - j as isize;
            // Even indices map to upsampled signal
            if raw_idx >= 0 && raw_idx % 2 == 0 {
                let orig_idx = (raw_idx / 2) as usize % n;
                acc += fj * coeff.get(orig_idx).copied().unwrap_or(0.0);
            } else if raw_idx < 0 {
                // Periodic wrap
                let wrapped = ((raw_idx % (2 * n as isize)) + 2 * n as isize) as usize;
                if wrapped.is_multiple_of(2) {
                    let orig_idx = wrapped / 2 % n;
                    acc += fj * coeff.get(orig_idx).copied().unwrap_or(0.0);
                }
            }
        }
        if let Some(slot) = out.get_mut(k) {
            *slot = acc;
        }
        let _ = up_len;
    }
    out
}

/// Single-level discrete wavelet transform using db4 filters with periodic extension.
/// Returns `(approximation_coeffs, detail_coeffs)`.
#[must_use]
pub fn dwt_1d(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    if signal.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let approx = convolve_downsample(signal, &DB4_LO);
    let detail = convolve_downsample(signal, &DB4_HI);
    (approx, detail)
}

/// Inverse single-level DWT via upsampling + convolution with synthesis filters.
/// Returns signal reconstructed to `approx.len() + detail.len()` samples.
#[must_use]
pub fn idwt_1d(approx: &[f64], detail: &[f64]) -> Vec<f64> {
    let out_len = approx.len() + detail.len();
    if out_len == 0 {
        return Vec::new();
    }
    let lo_part = upsample_convolve(approx, &DB4_LO_SYNTH, out_len);
    let hi_part = upsample_convolve(detail, &DB4_HI_SYNTH, out_len);
    lo_part
        .iter()
        .zip(hi_part.iter())
        .map(|(&l, &h)| l + h)
        .collect()
}

/// Multi-level DWT decomposition.
/// Returns `[cA_n, cD_n, cD_{n-1}, ..., cD_1]`:
/// - First element = coarsest approximation.
/// - Remaining elements = detail coefficients from coarsest to finest.
#[must_use]
pub fn wavedec(signal: &[f64], levels: usize) -> Vec<Vec<f64>> {
    if signal.is_empty() || levels == 0 {
        return vec![signal.to_vec()];
    }
    let mut result: Vec<Vec<f64>> = Vec::with_capacity(levels + 1);
    let mut current = signal.to_vec();

    // details collected from finest to coarsest; we reverse at end
    let mut details: Vec<Vec<f64>> = Vec::with_capacity(levels);

    for _ in 0..levels {
        if current.len() < 2 {
            break;
        }
        let (approx, detail) = dwt_1d(&current);
        details.push(detail);
        current = approx;
    }

    // result = [coarsest_approx, detail_coarsest, ..., detail_finest]
    result.push(current);
    for d in details.into_iter().rev() {
        result.push(d);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dwt_1d_basic() {
        let signal: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let (approx, detail) = dwt_1d(&signal);
        // Output should have ~8 coefficients each
        assert!(!approx.is_empty());
        assert!(!detail.is_empty());
        // approx + detail lengths ~ original length
        assert_eq!(approx.len() + detail.len(), signal.len());
    }

    #[test]
    fn test_idwt_1d_reconstruction_shape() {
        let signal: Vec<f64> = (0..8).map(|i| (i as f64).sin()).collect();
        let (approx, detail) = dwt_1d(&signal);
        let reconstructed = idwt_1d(&approx, &detail);
        // Reconstruction length should match original
        assert_eq!(reconstructed.len(), signal.len());
    }

    #[test]
    fn test_wavedec_levels() {
        let signal: Vec<f64> = (0..32).map(|i| (i as f64 * 0.5).sin()).collect();
        let decomp = wavedec(&signal, 3);
        // Should have 1 + 3 = 4 coefficient arrays
        assert_eq!(decomp.len(), 4);
        // First is coarsest approximation, should be shorter than signal
        let coarsest = decomp.first().map(|v| v.len()).unwrap_or(0);
        assert!(coarsest < signal.len());
    }

    #[test]
    fn test_db4_filter_energy() {
        // DB4 filters should have energy ~1
        let lo_energy: f64 = DB4_LO.iter().map(|&x| x * x).sum();
        assert!((lo_energy - 1.0).abs() < 0.01, "DB4_LO energy: {lo_energy}");
    }

    #[test]
    fn test_wavedec_empty() {
        let decomp = wavedec(&[], 3);
        assert_eq!(decomp.len(), 1);
        assert!(decomp.first().map(|v| v.is_empty()).unwrap_or(false));
    }
}
