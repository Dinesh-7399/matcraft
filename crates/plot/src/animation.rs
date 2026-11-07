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

//! SVG animation generator using `<animate>` keyframe animation.

use crate::PlotError;

/// A single animation frame: timestamp and (x, y) data points.
#[derive(Clone, Debug, PartialEq)]
pub struct AnimFrame {
    pub time_sec: f64,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// Render an SVG animation where each frame's polyline appears as a keyframe.
///
/// Uses a single `<polyline>` with `<animate attributeName="points" ...>` to
/// animate between frames. Auto-scales x/y across all frames to fit the viewport.
/// Returns an error if `frames` is empty.
pub fn render_svg_animation(
    frames: &[AnimFrame],
    width: u32,
    height: u32,
    title: &str,
) -> Result<String, PlotError> {
    if frames.is_empty() {
        return Err(PlotError::EmptyData);
    }

    let w = f64::from(width.max(200));
    let h = f64::from(height.max(150));
    let pad = 50.0_f64;
    let plot_w = w - 2.0 * pad;
    let plot_h = h - 2.0 * pad - 30.0; // extra bottom margin for title

    // Compute global x/y ranges across all frames
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for frame in frames {
        for &xv in &frame.x {
            if xv < min_x { min_x = xv; }
            if xv > max_x { max_x = xv; }
        }
        for &yv in &frame.y {
            if yv < min_y { min_y = yv; }
            if yv > max_y { max_y = yv; }
        }
    }

    // Guard degenerate ranges
    if !min_x.is_finite() || (max_x - min_x).abs() < 1e-12 {
        min_x = 0.0; max_x = 1.0;
    }
    if !min_y.is_finite() || (max_y - min_y).abs() < 1e-12 {
        min_y = -1.0; max_y = 1.0;
    }

    let x_range = max_x - min_x;
    let y_range = max_y - min_y;

    // Convert (x, y) data coordinates to SVG pixel coordinates
    let to_svg = |xv: f64, yv: f64| -> (f64, f64) {
        let px = pad + ((xv - min_x) / x_range) * plot_w;
        let py = pad + plot_h - ((yv - min_y) / y_range) * plot_h;
        (px, py)
    };

    // Build keyframe point strings for each frame
    let mut key_values: Vec<String> = Vec::with_capacity(frames.len());
    for frame in frames {
        let mut pts = String::new();
        let n_pts = frame.x.len().min(frame.y.len());
        for i in 0..n_pts {
            let xv = frame.x.get(i).copied().unwrap_or(0.0);
            let yv = frame.y.get(i).copied().unwrap_or(0.0);
            let (px, py) = to_svg(xv, yv);
            if !pts.is_empty() {
                pts.push(' ');
            }
            pts.push_str(&format!("{px:.2},{py:.2}"));
        }
        key_values.push(pts);
    }

    // Build key times (evenly spaced 0.0..1.0)
    let key_times: Vec<String> = {
        let n = frames.len();
        (0..n)
            .map(|i| {
                if n == 1 {
                    "0;1".to_string()
                } else {
                    format!("{:.4}", i as f64 / (n - 1) as f64)
                }
            })
            .collect()
    };

    // Compute total animation duration
    let dur = frames
        .last()
        .map(|f| f.time_sec)
        .unwrap_or(1.0)
        .max(0.001);

    // Build SVG
    let n_frames = frames.len();
    let first_pts = key_values.first().cloned().unwrap_or_default();
    let anim_values = key_values.join(";");
    let anim_key_times = key_times.join(";");

    let mut svg = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#11161d"/>
  <rect x="{pad}" y="{pad}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#3b4654" stroke-width="1"/>
"##,
        pad = pad,
        plot_w = plot_w,
        plot_h = plot_h,
    );

    // Title
    svg.push_str(&format!(
        "  <text x=\"{cx}\" y=\"24\" fill=\"#e6edf3\" font-family=\"monospace\" font-size=\"14\" text-anchor=\"middle\">{title}</text>\n",
        cx = w * 0.5,
    ));

    // Axes labels
    svg.push_str(&format!(
        "  <text x=\"{cx}\" y=\"{ty}\" fill=\"#8b949e\" font-family=\"monospace\" font-size=\"11\" text-anchor=\"middle\">x: [{min_x:.2}, {max_x:.2}]  y: [{min_y:.2}, {max_y:.2}]</text>\n",
        cx = w * 0.5,
        ty = pad + plot_h + 20.0,
        min_x = min_x,
        max_x = max_x,
        min_y = min_y,
        max_y = max_y,
    ));

    // Frame count annotation
    svg.push_str(&format!(
        "  <text x=\"{x}\" y=\"{y}\" fill=\"#58a6ff\" font-family=\"monospace\" font-size=\"10\">{n_frames} frames</text>\n",
        x = pad + 4.0,
        y = pad + 14.0,
    ));

    // Animated polyline
    svg.push_str(&format!(
        "  <polyline id=\"anim_line\" fill=\"none\" stroke=\"#58a6ff\" stroke-width=\"2\" points=\"{first_pts}\">\n"
    ));
    svg.push_str(&format!(
        "    <animate attributeName=\"points\" dur=\"{dur:.3}s\" repeatCount=\"indefinite\" calcMode=\"linear\"\n"
    ));
    svg.push_str(&format!(
        "      keyTimes=\"{anim_key_times}\"\n"
    ));
    svg.push_str(&format!(
        "      values=\"{anim_values}\"/>\n"
    ));
    svg.push_str("  </polyline>\n");
    svg.push_str("</svg>");

    Ok(svg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_svg_animation_basic() -> Result<(), PlotError> {
        let frames = vec![
            AnimFrame { time_sec: 0.0, x: vec![0.0, 1.0, 2.0], y: vec![0.0, 1.0, 0.0] },
            AnimFrame { time_sec: 0.5, x: vec![0.0, 1.0, 2.0], y: vec![0.5, 1.5, 0.5] },
            AnimFrame { time_sec: 1.0, x: vec![0.0, 1.0, 2.0], y: vec![1.0, 0.0, 1.0] },
        ];
        let svg = render_svg_animation(&frames, 400, 300, "Test Animation")?;
        assert!(svg.contains("<svg"), "Should contain SVG root element");
        assert!(svg.contains("<animate"), "Should contain animate element");
        assert!(svg.contains("Test Animation"), "Should contain title");
        assert!(svg.contains("3 frames"), "Should annotate frame count");
        assert!(svg.contains("attributeName=\"points\""), "Should animate points");
        Ok(())
    }

    #[test]
    fn test_render_svg_animation_empty_returns_error() {
        let result = render_svg_animation(&[], 400, 300, "Empty");
        assert!(result.is_err());
    }

    #[test]
    fn test_render_svg_animation_single_frame() -> Result<(), PlotError> {
        let frames = vec![
            AnimFrame { time_sec: 0.0, x: vec![0.0, 1.0], y: vec![0.0, 1.0] },
        ];
        let svg = render_svg_animation(&frames, 300, 200, "Single")?;
        assert!(svg.contains("<polyline"));
        Ok(())
    }

    #[test]
    fn test_render_svg_animation_scales_correctly() -> Result<(), PlotError> {
        // All x in [0, 10], all y in [-5, 5]
        let frames = vec![
            AnimFrame {
                time_sec: 0.0,
                x: vec![0.0, 5.0, 10.0],
                y: vec![-5.0, 0.0, 5.0],
            },
        ];
        let svg = render_svg_animation(&frames, 500, 400, "Scale Test")?;
        assert!(svg.contains("-5.00"));
        assert!(svg.contains("10.00"));
        Ok(())
    }
}
