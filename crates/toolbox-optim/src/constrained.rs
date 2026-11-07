use crate::OptimError;

/// Derivative-Free Nelder-Mead Downhill Simplex Optimizer (`fminsearch`).
/// Minimizes non-smooth or black-box objective functions $f: \mathbb{R}^n \to \mathbb{R}$ without derivatives!
pub fn fminsearch_nelder_mead<F>(
    x0: &[f64],
    tol: f64,
    max_iter: usize,
    objective: F,
) -> Result<(Vec<f64>, f64), OptimError>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Ok((Vec::new(), 0.0));
    }

    let mut simplex: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n + 1);
    simplex.push((x0.to_vec(), objective(x0)));
    for i in 0..n {
        let mut pt = x0.to_vec();
        if let Some(slot) = pt.get_mut(i) {
            let delta = if slot.abs() > 1e-4 {
                0.05 * (*slot)
            } else {
                0.0025
            };
            *slot += delta;
        }
        let val = objective(&pt);
        simplex.push((pt, val));
    }

    for _iter in 0..max_iter {
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        let best_f = simplex.first().map_or(0.0, |p| p.1);
        let worst_f = simplex.last().map_or(0.0, |p| p.1);
        if (worst_f - best_f).abs() <= tol {
            let best = if !simplex.is_empty() {
                simplex.swap_remove(0)
            } else {
                (x0.to_vec(), 0.0)
            };
            return Ok(best);
        }

        // Compute centroid of best n points (excluding worst at index n)
        let mut centroid = vec![0.0; n];
        for (pt, _) in simplex.iter().take(n) {
            for (c, &v) in centroid.iter_mut().zip(pt) {
                *c += v / (n as f64);
            }
        }

        let worst_pt = simplex
            .last()
            .map(|p| p.0.clone())
            .unwrap_or_else(|| vec![0.0; n]);
        let second_worst_f = simplex.get(n.saturating_sub(1)).map_or(worst_f, |p| p.1);

        // Reflection: x_r = c + 1.0 * (c - x_worst)
        let x_r: Vec<f64> = centroid
            .iter()
            .zip(&worst_pt)
            .map(|(&c, &w)| c + (c - w))
            .collect();
        let f_r = objective(&x_r);

        if f_r < best_f {
            // Expansion: x_e = c + 2.0 * (x_r - c)
            let x_e: Vec<f64> = centroid
                .iter()
                .zip(&x_r)
                .map(|(&c, &r)| c + 2.0 * (r - c))
                .collect();
            let f_e = objective(&x_e);
            if let Some(last) = simplex.last_mut() {
                *last = if f_e < f_r { (x_e, f_e) } else { (x_r, f_r) };
            }
        } else if f_r < second_worst_f {
            if let Some(last) = simplex.last_mut() {
                *last = (x_r, f_r);
            }
        } else {
            // Contraction
            let x_c: Vec<f64> = centroid
                .iter()
                .zip(&worst_pt)
                .map(|(&c, &w)| c + 0.5 * (w - c))
                .collect();
            let f_c = objective(&x_c);
            if f_c < worst_f {
                if let Some(last) = simplex.last_mut() {
                    *last = (x_c, f_c);
                }
            } else {
                // Shrink toward best point
                let best_pt = simplex
                    .first()
                    .map(|p| p.0.clone())
                    .unwrap_or_else(|| vec![0.0; n]);
                for item in simplex.iter_mut().skip(1) {
                    for (xi, &bi) in item.0.iter_mut().zip(&best_pt) {
                        *xi = bi + 0.5 * (*xi - bi);
                    }
                    item.1 = objective(&item.0);
                }
            }
        }
    }

    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
    let best = if !simplex.is_empty() {
        simplex.swap_remove(0)
    } else {
        (x0.to_vec(), 0.0)
    };
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fminsearch_minimizes_nonsmooth_l1_cone() -> Result<(), Box<dyn std::error::Error>> {
        // Minimize non-differentiable f(x, y) = |x - 3| + |y + 2|
        let (x_opt, f_opt) = fminsearch_nelder_mead(&[0.0, 0.0], 1e-7, 400, |v| {
            let x = v.first().copied().unwrap_or(0.0);
            let y = v.get(1).copied().unwrap_or(0.0);
            (x - 3.0).abs() + (y + 2.0).abs()
        })?;
        assert!((x_opt.first().copied().unwrap_or(0.0) - 3.0).abs() < 1e-3);
        assert!((x_opt.get(1).copied().unwrap_or(0.0) - (-2.0)).abs() < 1e-3);
        assert!(f_opt < 2e-3);
        Ok(())
    }
}
