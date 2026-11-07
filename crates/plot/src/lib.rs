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

pub mod animation;
pub mod charts;
pub mod root_locus_plot;

pub use animation::{render_svg_animation, AnimFrame};
pub use charts::{march_squares_contour, LineSegment2D};
pub use root_locus_plot::{render_root_locus_svg, LocusPlotData};

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Error type for plot operations.
#[derive(Clone, Debug, PartialEq)]
pub enum PlotError {
    /// No data provided to render.
    EmptyData,
    /// Generic rendering failure.
    RenderError(String),
}

/// Largest-Triangle-Three-Buckets (LTTB) downsampling algorithm for high-performance plotting.
/// Reduces `x` and `y` series of size $N$ down to `threshold` points while preserving visual extrema.
#[must_use]
pub fn lttb_downsample(x: &[f64], y: &[f64], threshold: usize) -> (Vec<f64>, Vec<f64>) {
    let n = x.len().min(y.len());
    if threshold >= n || threshold <= 2 || n <= 2 {
        let x_out = x.iter().take(n).copied().collect();
        let y_out = y.iter().take(n).copied().collect();
        return (x_out, y_out);
    }

    let mut sampled_x = Vec::with_capacity(threshold);
    let mut sampled_y = Vec::with_capacity(threshold);

    // Always include the very first point
    let first_x = x.first().copied().unwrap_or(0.0);
    let first_y = y.first().copied().unwrap_or(0.0);
    sampled_x.push(first_x);
    sampled_y.push(first_y);

    let bucket_size = ((n - 2) as f64) / ((threshold - 2) as f64);
    let mut a_idx = 0usize;

    for i in 0..(threshold - 2) {
        // Range of bucket C (the next bucket to average)
        let c_start = (((i + 1) as f64) * bucket_size).floor() as usize + 1;
        let c_end = ((((i + 2) as f64) * bucket_size).floor() as usize + 1).min(n);

        let mut avg_x = 0.0;
        let mut avg_y = 0.0;
        let count_c = (c_end.saturating_sub(c_start)).max(1);

        for c_idx in c_start..c_end {
            avg_x += x.get(c_idx).copied().unwrap_or(0.0);
            avg_y += y.get(c_idx).copied().unwrap_or(0.0);
        }
        avg_x /= count_c as f64;
        avg_y /= count_c as f64;

        // Range of bucket B (the current bucket to select from)
        let b_start = ((i as f64) * bucket_size).floor() as usize + 1;
        let b_end = (((i + 1) as f64) * bucket_size).floor() as usize + 1;

        let ax = x.get(a_idx).copied().unwrap_or(0.0);
        let ay = y.get(a_idx).copied().unwrap_or(0.0);

        let mut max_area = -1.0;
        let mut max_idx = b_start;

        for b_idx in b_start..b_end.min(n) {
            let bx = x.get(b_idx).copied().unwrap_or(0.0);
            let by = y.get(b_idx).copied().unwrap_or(0.0);

            // Triangle area: 0.5 * |(Ax - Cx)(By - Ay) - (Ax - Bx)(Cy - Ay)|
            let area = ((ax - avg_x) * (by - ay) - (ax - bx) * (avg_y - ay)).abs() * 0.5;
            if area > max_area {
                max_area = area;
                max_idx = b_idx;
            }
        }

        let sel_x = x.get(max_idx).copied().unwrap_or(0.0);
        let sel_y = y.get(max_idx).copied().unwrap_or(0.0);
        sampled_x.push(sel_x);
        sampled_y.push(sel_y);
        a_idx = max_idx;
    }

    // Always include the very last point
    let last_x = x.get(n.saturating_sub(1)).copied().unwrap_or(0.0);
    let last_y = y.get(n.saturating_sub(1)).copied().unwrap_or(0.0);
    sampled_x.push(last_x);
    sampled_y.push(last_y);

    (sampled_x, sampled_y)
}

/// Viridis colormap approximation for normalizing value $t \in [0, 1]$ into RGB bytes.
#[must_use]
pub fn viridis_color(t: f64) -> [u8; 3] {
    let c = t.clamp(0.0, 1.0);
    // Smooth polynomial approximation of Viridis color curve
    let r = (-0.002 + 0.99 * c - 0.52 * c * c + 0.44 * c * c * c).clamp(0.0, 1.0);
    let g = (0.01 + 0.77 * c + 0.35 * c * c - 0.28 * c * c * c).clamp(0.0, 1.0);
    let b = (0.33 + 0.88 * c - 1.85 * c * c + 0.81 * c * c * c).clamp(0.0, 1.0);
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

/// 3D Surface / Mesh scene representation (`surf(X, Y, Z)` / `mesh(X, Y, Z)`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceMesh3D {
    pub title: String,
    pub x_coords: Vec<f64>,
    pub y_coords: Vec<f64>,
    pub z_grid: Vec<Vec<f64>>,
}

impl SurfaceMesh3D {
    #[must_use]
    pub fn new(title: &str, x: Vec<f64>, y: Vec<f64>, z: Vec<Vec<f64>>) -> Self {
        Self {
            title: title.to_string(),
            x_coords: x,
            y_coords: y,
            z_grid: z,
        }
    }

    /// Renders an isometric 3D surface view to standalone SVG with Viridis shaded facets.
    #[must_use]
    pub fn render_to_svg(&self, width: u32, height: u32) -> String {
        let w = f64::from(width.max(300));
        let h = f64::from(height.max(200));

        let nx = self.x_coords.len();
        let ny = self.y_coords.len();
        if nx < 2 || ny < 2 {
            return format!("<svg width=\"{width}\" height=\"{height}\"></svg>");
        }

        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;
        for row in &self.z_grid {
            for &zv in row {
                min_z = min_z.min(zv);
                max_z = max_z.max(zv);
            }
        }
        let z_range = if (max_z - min_z).abs() < 1e-12 { 1.0 } else { max_z - min_z };

        // Isometric projection angles
        let cos_iso = (PI / 6.0).cos();
        let sin_iso = (PI / 6.0).sin();

        let cx = w * 0.5;
        let cy = h * 0.55;
        let scale_xy = (w.min(h) * 0.35) / ((nx.max(ny) as f64) * 0.7);
        let scale_z = (h * 0.25) / z_range;

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\n\
             <rect width=\"100%\" height=\"100%\" fill=\"#11161d\"/>\n\
             <text x=\"{}\" y=\"24\" fill=\"#e6edf3\" font-family=\"monospace\" font-size=\"14\" text-anchor=\"middle\">{}</text>\n",
            w * 0.5,
            self.title
        );

        // Render surface quads back-to-front
        for i in 0..(nx - 1) {
            for j in 0..(ny - 1) {
                let x0 = (i as f64) - (nx as f64) * 0.5;
                let x1 = ((i + 1) as f64) - (nx as f64) * 0.5;
                let y0 = (j as f64) - (ny as f64) * 0.5;
                let y1 = ((j + 1) as f64) - (ny as f64) * 0.5;

                let z00 = self.z_grid.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
                let z10 = self.z_grid.get(i + 1).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
                let z11 = self.z_grid.get(i + 1).and_then(|r| r.get(j + 1)).copied().unwrap_or(0.0);
                let z01 = self.z_grid.get(i).and_then(|r| r.get(j + 1)).copied().unwrap_or(0.0);

                let avg_z = (z00 + z10 + z11 + z01) * 0.25;
                let norm_z = (avg_z - min_z) / z_range;
                let [r, g, b] = viridis_color(norm_z);

                // Project 4 vertices
                let p00_x = cx + (x0 - y0) * cos_iso * scale_xy;
                let p00_y = cy + (x0 + y0) * sin_iso * scale_xy - (z00 - min_z) * scale_z;

                let p10_x = cx + (x1 - y0) * cos_iso * scale_xy;
                let p10_y = cy + (x1 + y0) * sin_iso * scale_xy - (z10 - min_z) * scale_z;

                let p11_x = cx + (x1 - y1) * cos_iso * scale_xy;
                let p11_y = cy + (x1 + y1) * sin_iso * scale_xy - (z11 - min_z) * scale_z;

                let p01_x = cx + (x0 - y1) * cos_iso * scale_xy;
                let p01_y = cy + (x0 + y1) * sin_iso * scale_xy - (z01 - min_z) * scale_z;

                svg.push_str(&format!(
                    "  <polygon points=\"{p00_x:.1},{p00_y:.1} {p10_x:.1},{p10_y:.1} {p11_x:.1},{p11_y:.1} {p01_x:.1},{p01_y:.1}\" \
                     fill=\"rgb({r},{g},{b})\" stroke=\"#1c2128\" stroke-width=\"0.5\"/>\n"
                ));
            }
        }

        svg.push_str("</svg>");
        svg
    }
}

/// Dual-subplot Bode frequency response renderer (`bode(sys)`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodePlot {
    pub title: String,
    pub freqs_rad_s: Vec<f64>,
    pub mag_db: Vec<f64>,
    pub phase_deg: Vec<f64>,
}

impl BodePlot {
    #[must_use]
    pub fn new(title: &str, freqs: Vec<f64>, mag_db: Vec<f64>, phase_deg: Vec<f64>) -> Self {
        Self {
            title: title.to_string(),
            freqs_rad_s: freqs,
            mag_db,
            phase_deg,
        }
    }

    /// Renders dual magnitude and phase subplots to standalone SVG.
    #[must_use]
    pub fn render_to_svg(&self, width: u32, height: u32) -> String {
        let w = f64::from(width.max(300));
        let h = f64::from(height.max(250));
        let pad = 40.0;
        let sub_h = (h - 3.0 * pad) * 0.5;
        let plot_w = w - 2.0 * pad;

        let n = self.freqs_rad_s.len();
        if n == 0 {
            return format!("<svg width=\"{width}\" height=\"{height}\"></svg>");
        }

        let f_min = self.freqs_rad_s.first().copied().unwrap_or(0.1).max(1e-6).log10();
        let f_max = self.freqs_rad_s.last().copied().unwrap_or(100.0).max(1e-5).log10();
        let f_span = if (f_max - f_min).abs() < 1e-6 { 1.0 } else { f_max - f_min };

        let mut min_mag = f64::INFINITY;
        let mut max_mag = f64::NEG_INFINITY;
        for &m in &self.mag_db {
            min_mag = min_mag.min(m);
            max_mag = max_mag.max(m);
        }
        let mag_span = if (max_mag - min_mag).abs() < 1e-6 { 10.0 } else { max_mag - min_mag };

        let mut min_phase = f64::INFINITY;
        let mut max_phase = f64::NEG_INFINITY;
        for &p in &self.phase_deg {
            min_phase = min_phase.min(p);
            max_phase = max_phase.max(p);
        }
        let phase_span = if (max_phase - min_phase).abs() < 1e-6 { 45.0 } else { max_phase - min_phase };

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\n\
             <rect width=\"100%\" height=\"100%\" fill=\"#11161d\"/>\n\
             <text x=\"{}\" y=\"22\" fill=\"#e6edf3\" font-family=\"monospace\" font-size=\"14\" text-anchor=\"middle\">{} - Bode Diagram</text>\n",
            w * 0.5,
            self.title
        );

        // Subplot 1: Magnitude (dB)
        let y_top_mag = pad;
        svg.push_str(&format!(
            "  <rect x=\"{pad}\" y=\"{y_top_mag}\" width=\"{plot_w}\" height=\"{sub_h}\" fill=\"none\" stroke=\"#3b4654\" stroke-width=\"1\"/>\n\
             <text x=\"{}\" y=\"{}\" fill=\"#8b949e\" font-family=\"monospace\" font-size=\"11\">Magnitude (dB)</text>\n",
            pad + 6.0,
            y_top_mag + 16.0
        ));

        let mut mag_pts = String::new();
        for (i, &f) in self.freqs_rad_s.iter().enumerate() {
            let m = self.mag_db.get(i).copied().unwrap_or(0.0);
            let lf = f.max(1e-6).log10();
            let px = pad + ((lf - f_min) / f_span) * plot_w;
            let py = y_top_mag + sub_h - ((m - min_mag) / mag_span) * sub_h;
            mag_pts.push_str(&format!("{px:.1},{py:.1} "));
        }
        svg.push_str(&format!(
            "  <polyline fill=\"none\" stroke=\"#58a6ff\" stroke-width=\"2\" points=\"{mag_pts}\"/>\n"
        ));

        // Subplot 2: Phase (deg)
        let y_top_phase = pad * 2.0 + sub_h;
        svg.push_str(&format!(
            "  <rect x=\"{pad}\" y=\"{y_top_phase}\" width=\"{plot_w}\" height=\"{sub_h}\" fill=\"none\" stroke=\"#3b4654\" stroke-width=\"1\"/>\n\
             <text x=\"{}\" y=\"{}\" fill=\"#8b949e\" font-family=\"monospace\" font-size=\"11\">Phase (deg)</text>\n",
            pad + 6.0,
            y_top_phase + 16.0
        ));

        let mut phase_pts = String::new();
        for (i, &f) in self.freqs_rad_s.iter().enumerate() {
            let ph = self.phase_deg.get(i).copied().unwrap_or(0.0);
            let lf = f.max(1e-6).log10();
            let px = pad + ((lf - f_min) / f_span) * plot_w;
            let py = y_top_phase + sub_h - ((ph - min_phase) / phase_span) * sub_h;
            phase_pts.push_str(&format!("{px:.1},{py:.1} "));
        }
        svg.push_str(&format!(
            "  <polyline fill=\"none\" stroke=\"#3fb950\" stroke-width=\"2\" points=\"{phase_pts}\"/>\n\
             </svg>"
        ));

        svg
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LineSeries2D {
    pub label: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub color_rgb: [u8; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FigureScene {
    pub id: usize,
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub series: Vec<LineSeries2D>,
}

impl FigureScene {
    #[must_use]
    pub fn new(id: usize, title: &str) -> Self {
        Self {
            id,
            title: title.to_string(),
            x_label: "x".to_string(),
            y_label: "y".to_string(),
            series: Vec::new(),
        }
    }

    /// Renders the scene graph deterministically to a standalone SVG string (headless oracle).
    #[must_use]
    pub fn render_to_svg(&self, width: u32, height: u32) -> String {
        let w = f64::from(width.max(200));
        let h = f64::from(height.max(150));
        let pad = 40.0;
        let plot_w = w - 2.0 * pad;
        let plot_h = h - 2.0 * pad;

        let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
        for s in &self.series {
            for &xv in &s.x {
                min_x = min_x.min(xv);
                max_x = max_x.max(xv);
            }
            for &yv in &s.y {
                min_y = min_y.min(yv);
                max_y = max_y.max(yv);
            }
        }
        if !min_x.is_finite() || (max_x - min_x).abs() < 1e-12 {
            min_x = 0.0;
            max_x = 1.0;
        }
        if !min_y.is_finite() || (max_y - min_y).abs() < 1e-12 {
            min_y = -1.0;
            max_y = 1.0;
        }

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\n"
        );
        svg.push_str(&format!(
            "  <rect width=\"100%\" height=\"100%\" fill=\"#11161d\"/>\n  <rect x=\"{pad}\" y=\"{pad}\" width=\"{plot_w}\" height=\"{plot_h}\" fill=\"none\" stroke=\"#3b4654\" stroke-width=\"1\"/>\n"
        ));
        svg.push_str(&format!(
            "  <text x=\"{}\" y=\"24\" fill=\"#e6edf3\" font-family=\"monospace\" font-size=\"14\" text-anchor=\"middle\">{}</text>\n",
            w * 0.5,
            self.title
        ));

        for s in &self.series {
            let mut pts = String::new();
            for (&xv, &yv) in s.x.iter().zip(&s.y) {
                let px = pad + ((xv - min_x) / (max_x - min_x)) * plot_w;
                let py = pad + plot_h - ((yv - min_y) / (max_y - min_y)) * plot_h;
                pts.push_str(&format!("{px:.2},{py:.2} "));
            }
            let [r, g, b] = s.color_rgb;
            svg.push_str(&format!(
                "  <polyline fill=\"none\" stroke=\"rgb({r},{g},{b})\" stroke-width=\"2\" points=\"{pts}\"/>\n"
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    /// Deterministic software RGBA buffer rasterizer (`width * height * 4` bytes) for headless verification.
    #[must_use]
    #[allow(clippy::chunks_exact_to_as_chunks)]
    pub fn render_to_rgba(&self, width: usize, height: usize) -> Vec<u8> {
        let w = width.clamp(16, 4096);
        let h = height.clamp(16, 4096);
        let mut buf = vec![20u8; w.saturating_mul(h).saturating_mul(4)];
        for px in buf.chunks_exact_mut(4) {
            if let Some(a) = px.get_mut(3) {
                *a = 255;
            }
        }
        for s in &self.series {
            for (idx, &yv) in s.y.iter().enumerate() {
                let x_px = (idx.saturating_mul(w - 1)) / s.y.len().max(1);
                let norm_y = ((yv + 1.0) * 0.5).clamp(0.0, 1.0);
                let y_px = ((1.0 - norm_y) * ((h - 1) as f64)) as usize;
                let base = (y_px.saturating_mul(w).saturating_add(x_px)).saturating_mul(4);
                if let Some(slice) = buf.get_mut(base..base.saturating_add(4)) {
                    slice.copy_from_slice(&[s.color_rgb[0], s.color_rgb[1], s.color_rgb[2], 255]);
                }
            }
        }
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lttb_downsampling_preserves_extrema() {
        let n = 1000;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for i in 0..n {
            let t = (i as f64) * 0.01;
            xs.push(t);
            ys.push(t.sin());
        }

        let (dx, dy) = lttb_downsample(&xs, &ys, 50);
        assert_eq!(dx.len(), 50);
        assert_eq!(dy.len(), 50);
        assert_eq!(dx.first(), xs.first());
        assert_eq!(dx.last(), xs.last());
    }

    #[test]
    fn test_surface_and_bode_svg_renderers() {
        let x = vec![-1.0, 0.0, 1.0];
        let y = vec![-1.0, 0.0, 1.0];
        let z = vec![
            vec![1.0, 0.5, 1.0],
            vec![0.5, 0.0, 0.5],
            vec![1.0, 0.5, 1.0],
        ];
        let surf = SurfaceMesh3D::new("Saddle", x, y, z);
        let svg_surf = surf.render_to_svg(400, 300);
        assert!(svg_surf.contains("<polygon"));

        let bode = BodePlot::new(
            "Filter",
            vec![0.1, 1.0, 10.0],
            vec![0.0, -3.0, -20.0],
            vec![0.0, -45.0, -90.0],
        );
        let svg_bode = bode.render_to_svg(400, 300);
        assert!(svg_bode.contains("Magnitude (dB)"));
        assert!(svg_bode.contains("Phase (deg)"));
    }
}
