use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LineSegment2D {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

/// Marching Squares Iso-Contour Extractor (`contour(Z, level)`).
/// Extracts sub-cell interpolated iso-value line segments from a `rows x cols` scalar field `z_grid`!
#[must_use]
pub fn march_squares_contour(
    rows: usize,
    cols: usize,
    z_grid: &[f64],
    iso_level: f64,
) -> Vec<LineSegment2D> {
    let mut segments = Vec::new();
    if rows < 2 || cols < 2 || z_grid.len() < rows.saturating_mul(cols) {
        return segments;
    }

    for r in 0..(rows - 1) {
        for c in 0..(cols - 1) {
            let z00 = z_grid.get(r * cols + c).copied().unwrap_or(0.0);
            let z10 = z_grid.get(r * cols + (c + 1)).copied().unwrap_or(0.0);
            let z11 = z_grid.get((r + 1) * cols + (c + 1)).copied().unwrap_or(0.0);
            let z01 = z_grid.get((r + 1) * cols + c).copied().unwrap_or(0.0);

            let mut mask = 0u8;
            if z00 >= iso_level {
                mask |= 1;
            }
            if z10 >= iso_level {
                mask |= 2;
            }
            if z11 >= iso_level {
                mask |= 4;
            }
            if z01 >= iso_level {
                mask |= 8;
            }
            if mask == 0 || mask == 15 {
                continue;
            }

            let x = c as f64;
            let y = r as f64;
            let bottom = (x + interp_frac(z00, z10, iso_level), y);
            let right = (x + 1.0, y + interp_frac(z10, z11, iso_level));
            let top = (x + interp_frac(z01, z11, iso_level), y + 1.0);
            let left = (x, y + interp_frac(z00, z01, iso_level));

            match mask {
                1 | 14 => segments.push(seg(left, bottom)),
                2 | 13 => segments.push(seg(bottom, right)),
                4 | 11 => segments.push(seg(right, top)),
                8 | 7 => segments.push(seg(top, left)),
                3 | 12 => segments.push(seg(left, right)),
                6 | 9 => segments.push(seg(bottom, top)),
                5 => {
                    segments.push(seg(left, top));
                    segments.push(seg(bottom, right));
                }
                10 => {
                    segments.push(seg(left, bottom));
                    segments.push(seg(right, top));
                }
                _ => {}
            }
        }
    }
    segments
}

fn interp_frac(za: f64, zb: f64, level: f64) -> f64 {
    let denom = zb - za;
    if denom.abs() < 1e-15 {
        0.5
    } else {
        ((level - za) / denom).clamp(0.0, 1.0)
    }
}

const fn seg(a: (f64, f64), b: (f64, f64)) -> LineSegment2D {
    LineSegment2D {
        x0: a.0,
        y0: a.1,
        x1: b.0,
        y1: b.1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marching_squares_extracts_circle_contour() {
        let n = 11;
        let mut grid = Vec::with_capacity(n * n);
        for r in 0..n {
            for c in 0..n {
                let dx = (c as f64) - 5.0;
                let dy = (r as f64) - 5.0;
                grid.push(dx * dx + dy * dy);
            }
        }
        let segs = march_squares_contour(n, n, &grid, 9.0); // radius = 3 circle
        assert!(segs.len() >= 16);
    }
}
