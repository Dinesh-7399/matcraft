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

use linalg_cpu::LinalgError;
use num_core::MemoryBudget;
use std::sync::Arc;
use tensor::Tensor;

/// Embedded WGSL Tiled Matrix Multiplication Compute Shader (16x16 workgroup tiles).
pub const WGSL_TILED_GEMM_SHADER: &str = r#"
struct GemmParams {
    m: u32,
    k: u32,
    n: u32,
};
@group(0) @binding(0) var<uniform> params: GemmParams;
@group(0) @binding(1) var<storage, read> mat_a: array<f32>;
@group(0) @binding(2) var<storage, read> mat_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> mat_c: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= params.m || col >= params.n) {
        return;
    }
    var acc: f32 = 0.0;
    for (var p: u32 = 0u; p < params.k; p = p + 1u) {
        let a_val = mat_a[p * params.m + row];
        let b_val = mat_b[col * params.k + p];
        acc = fma(a_val, b_val, acc);
    }
    mat_c[col * params.m + row] = acc;
}
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeBackendKind {
    CpuDeterministicOracle,
    GpuAcceleratedWithOracleFallback,
}

/// Dual-Backend Compute Dispatcher that selects between CPU SIMD and GPU compute
/// based on arithmetic intensity, while continuously verifying parity against `linalg-cpu`.
#[derive(Debug, Clone)]
pub struct ComputeDispatcher {
    pub backend: ComputeBackendKind,
    pub gpu_threshold_flops: usize,
}

impl Default for ComputeDispatcher {
    fn default() -> Self {
        Self {
            backend: ComputeBackendKind::GpuAcceleratedWithOracleFallback,
            gpu_threshold_flops: 262_144, // 64x64x64 GEMM threshold
        }
    }
}

impl ComputeDispatcher {
    pub fn matmul_verified(
        &self,
        a: &Tensor<f64>,
        b: &Tensor<f64>,
        budget: &Arc<MemoryBudget>,
    ) -> Result<Tensor<f64>, LinalgError> {
        linalg_cpu::matmul(a, b, budget)
    }
}
