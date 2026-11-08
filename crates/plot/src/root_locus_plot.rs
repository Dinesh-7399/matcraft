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

//! Evans Root Locus and Pole-Zero SVG Charting.
//! Generates engineering-grade SVG diagrams with complex s-plane axes,
//! pole crosses ('x'), zero circles ('o'), and locus trajectory branches.

use crate::PlotError;
use num_core::Complex64;

/// Data model for rendering root locus plots.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LocusPlotData {
    /// Open-loop poles (plotted as crosses 'x').
    pub open_loop_poles: Vec<Complex64>,
    /// Open-loop zeros (plotted as circles 'o').
    pub open_loop_zeros: Vec<Complex64>,
    /// Trajectory branches (each branch is a sequence of pole positions as gain sweeps).
    pub branches: Vec<Vec<Complex64>>,
    /// Asymptote real centroid (where dashed asymptotes radiate from).
    pub centroid_real: Option<f64>,
    /// Asymptote angles in radians.
    pub asymptote_angles_rad: Vec<f64>,
}

/// Renders a deterministic SVG root locus diagram.
pub fn render_root_locus_svg(
    data: &LocusPlotData,
    width: u32,
    height: u32,
    title: &str,
) -> Result<String, PlotError> {
    if width < 100 || height < 100 {
        return Err(PlotError::RenderError("Dimensions too small".into()));
    }

    let w = width as f64;
    let h = height as f64;
    let pad = 50.0;
    let plot_w = w - 2.0 * pad;
    let plot_h = h - 2.0 * pad;

    // Collect all coordinates to compute bounding box
    let mut min_sigma = 0.0f64;
    let mut max_sigma = 0.0f64;
    let mut min_omega = 0.0f64;
    let mut max_omega = 0.0f64;

    let mut update_bounds = |c: Complex64| {
        if c.re < min_sigma { min_sigma = c.re; }
        if c.re > max_sigma { max_sigma = c.re; }
        if c.im < min_omega { min_omega = c.im; }
        if c.im > max_omega { max_omega = c.im; }
    };

    for &p in &data.open_loop_poles { update_bounds(p); }
    for &z in &data.open_loop_zeros { update_bounds(z); }
    for br in &data.branches {
        for &pt in br { update_bounds(pt); }
    }
    if let Some(c) = data.centroid_real {
        update_bounds(Complex64::new(c, 0.0));
    }

    // Add padding to margins and ensure non-zero span
    let span_sigma = (max_sigma - min_sigma).max(2.0);
    let span_omega = (max_omega - min_omega).max(2.0);
    min_sigma -= 0.15 * span_sigma;
    max_sigma += 0.15 * span_sigma;
    min_omega -= 0.15 * span_omega;
    max_omega += 0.15 * span_omega;

    let range_sigma = (max_sigma - min_sigma).max(0.001);
    let range_omega = (max_omega - min_omega).max(0.001);

    // Map s-plane (sigma, omega) to canvas (px, py)
    let map_pt = |c: Complex64| -> (f64, f64) {
        let px = pad + (c.re - min_sigma) / range_sigma * plot_w;
        let py = pad + (max_omega - c.im) / range_omega * plot_h;
        (px, py)
    };

    let mut svg = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#11161d"/>
  <rect x="{pad}" y="{pad}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#3b4654" stroke-width="1"/>
"##,
        pad = pad,
        plot_w = plot_w,
        plot_h = plot_h,
    );

    // Draw real axis (omega = 0) and imaginary axis (sigma = 0)
    let (ax_zero_x, _) = map_pt(Complex64::new(0.0, 0.0));
    let (_, ax_zero_y) = map_pt(Complex64::new(0.0, 0.0));

    // Stability boundary (imaginary axis sigma = 0)
    if (pad..=(pad + plot_w)).contains(&ax_zero_x) {
        svg.push_str(&format!(
            "  <line x1=\"{x}\" y1=\"{y1}\" x2=\"{x}\" y2=\"{y2}\" stroke=\"#8b949e\" stroke-width=\"1.5\" stroke-dasharray=\"4,4\"/>\n",
            x = ax_zero_x,
            y1 = pad,
            y2 = pad + plot_h,
        ));
    }

    // Real axis (omega = 0)
    if (pad..=(pad + plot_h)).contains(&ax_zero_y) {
        svg.push_str(&format!(
            "  <line x1=\"{x1}\" y1=\"{y}\" x2=\"{x2}\" y2=\"{y}\" stroke=\"#3b4654\" stroke-width=\"1\"/>\n",
            x1 = pad,
            x2 = pad + plot_w,
            y = ax_zero_y,
        ));
    }

    // Draw Asymptotes if present
    if let (Some(cent), false) = (data.centroid_real, data.asymptote_angles_rad.is_empty()) {
        let (cx, cy) = map_pt(Complex64::new(cent, 0.0));
        let r_ray = plot_w.max(plot_h);
        for &angle in &data.asymptote_angles_rad {
            let target_s = Complex64::new(cent + r_ray * angle.cos(), r_ray * angle.sin());
            let (tx, ty) = map_pt(target_s);
            svg.push_str(&format!(
                "  <line x1=\"{cx:.1}\" y1=\"{cy:.1}\" x2=\"{tx:.1}\" y2=\"{ty:.1}\" stroke=\"#6e7681\" stroke-width=\"1\" stroke-dasharray=\"2,4\"/>\n"
            ));
        }
    }

    // Draw locus branches
    for br in &data.branches {
        if br.len() < 2 {
            continue;
        }
        let mut pts_str = String::new();
        for &c in br {
            let (px, py) = map_pt(c);
            if !pts_str.is_empty() {
                pts_str.push(' ');
            }
            pts_str.push_str(&format!("{px:.1},{py:.1}"));
        }
        svg.push_str(&format!(
            "  <polyline fill=\"none\" stroke=\"#58a6ff\" stroke-width=\"2\" points=\"{pts_str}\"/>\n"
        ));
    }

    // Draw open-loop poles ('x' mark)
    for &pole in &data.open_loop_poles {
        let (px, py) = map_pt(pole);
        let arm = 5.0;
        svg.push_str(&format!(
            "  <line x1=\"{x1:.1}\" y1=\"{y1:.1}\" x2=\"{x2:.1}\" y2=\"{y2:.1}\" stroke=\"#ff7b72\" stroke-width=\"2\"/>\n",
            x1 = px - arm, y1 = py - arm, x2 = px + arm, y2 = py + arm
        ));
        svg.push_str(&format!(
            "  <line x1=\"{x1:.1}\" y1=\"{y1:.1}\" x2=\"{x2:.1}\" y2=\"{y2:.1}\" stroke=\"#ff7b72\" stroke-width=\"2\"/>\n",
            x1 = px - arm, y1 = py + arm, x2 = px + arm, y2 = py - arm
        ));
    }

    // Draw open-loop zeros ('o' mark)
    for &zero in &data.open_loop_zeros {
        let (px, py) = map_pt(zero);
        let radius = 4.5;
        svg.push_str(&format!(
            "  <circle cx=\"{px:.1}\" cy=\"{py:.1}\" r=\"{radius}\" fill=\"none\" stroke=\"#3fb950\" stroke-width=\"2\"/>\n"
        ));
    }

    // Title and Labels
    let escaped_title = crate::escape_xml(title);
    svg.push_str(&format!(
        "  <text x=\"{cx}\" y=\"24\" fill=\"#e6edf3\" font-family=\"monospace\" font-size=\"14\" text-anchor=\"middle\">{escaped_title}</text>\n",
        cx = w * 0.5,
    ));
    svg.push_str(&format!(
        "  <text x=\"{x}\" y=\"{y}\" fill=\"#8b949e\" font-family=\"monospace\" font-size=\"11\" text-anchor=\"end\">Real Axis (sigma)</text>\n",
        x = pad + plot_w,
        y = pad + plot_h + 20.0,
    ));
    svg.push_str(&format!(
        "  <text x=\"{x}\" y=\"{y}\" fill=\"#8b949e\" font-family=\"monospace\" font-size=\"11\" text-anchor=\"start\">Imaginary Axis (j_omega)</text>\n",
        x = pad,
        y = pad - 10.0,
    ));

    svg.push_str("</svg>\n");
    Ok(svg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_root_locus_svg_basic() -> Result<(), PlotError> {
        let data = LocusPlotData {
            open_loop_poles: vec![Complex64::new(-1.0, 2.0), Complex64::new(-1.0, -2.0)],
            open_loop_zeros: vec![Complex64::new(-2.0, 0.0)],
            branches: vec![
                vec![Complex64::new(-1.0, 2.0), Complex64::new(-1.5, 1.0), Complex64::new(-2.0, 0.0)],
                vec![Complex64::new(-1.0, -2.0), Complex64::new(-1.5, -1.0), Complex64::new(-2.0, 0.0)],
            ],
            centroid_real: Some(-0.5),
            asymptote_angles_rad: vec![std::f64::consts::PI],
        };

        let svg = render_root_locus_svg(&data, 600, 400, "Root Locus: G(s) = (s+2)/(s^2+2s+5)")?;
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Root Locus"));
        assert!(svg.contains("stroke=\"#ff7b72\"")); // poles
        assert!(svg.contains("stroke=\"#3fb950\"")); // zeros
        assert!(svg.contains("polyline"));          // locus branch
        Ok(())
    }

    #[test]
    fn test_render_root_locus_svg_dimensions() {
        let data = LocusPlotData::default();
        let err = render_root_locus_svg(&data, 50, 50, "Tiny");
        assert!(err.is_err());
    }
}
