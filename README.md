# MatCraft 🚀

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Rust: 2021](https://img.shields.io/badge/Rust-2021%20Edition-orange.svg)](https://www.rust-lang.org)
[![Safety: #![forbid(unsafe_code)]](https://img.shields.io/badge/Safety-%23!%5Bforbid(unsafe__code)%5D-brightgreen.svg)](#)
[![Zero-Panic Architecture](https://img.shields.io/badge/Reliability-Zero--Panic-purple.svg)](#)

**MatCraft** is an industrial-grade, 100% pure Rust technical computing and simulation environment designed as a memory-safe, deterministic, modern successor to MATLAB® and Simulink®.

---

## 🏛️ Core Architectural Pillars & Golden Rules

Every crate in MatCraft strictly enforces zero-compromise engineering rules:

1. **`#![forbid(unsafe_code)]`**: Every single crate is 100% memory-safe.
2. **Deterministic Error Handling**: Clippy denies `unwrap`, `expect`, `panic`, unchecked `indexing_slicing`, `todo`, and `unimplemented`. Every operation returns structured `Result<T, E>`.
3. **Strict Downward-Only Layering (L0 → L7)**: Guaranteed zero circular dependencies across the entire workspace.
4. **Copy-on-Write (CoW) Memory Model**: `Tensor<T>` references are `Arc`-backed for instant zero-copy workspace snapshots and undo/redo histories.
5. **Compile-Time Physical Units**: Native SI unit tracking and verification across algebraic expressions (`5_m / 2_s = 2.5_m/s`).

---

## 📐 Layer Hierarchy (L0 → L7)

| Layer | Crates | Description |
|:------|:-------|:------------|
| **L0** | `num-core`, `tensor`, `format-mat` | Foundational numerical types (`Complex64`, `Dual64`), Copy-on-Write N-D strided tensors, heterogeneous Cell/Struct containers, Level 5 & 7.3 HDF5/chunked `.mat` serialization. |
| **L1** | `autodiff`, `lang-m` | Forward/reverse automatic differentiation, exact Symbolic Computer Algebra (CAS), Register Bytecode VM, and full `.m` script parser & AST interpreter. |
| **L2** | `linalg-cpu`, `compute-gpu` | Dense and sparse CPU reference linear algebra (LU, QR, Cholesky, SVD, Francis QR eigenvalues, Arnoldi sparse `eigs`, Krylov PCG/GMRES/BiCGSTAB, matrix exponential `expm`, Riccati/Lyapunov solvers) and WGSL/WGPU compute kernels. |
| **L3** | `sim-engine`, `toolbox-control`, `toolbox-signal`, `toolbox-optim` | Simulink-grade acausal physical modeling (MNA), ODE45/ODE15s solvers, Stateflow FSMs, Pantelides DAE index reduction, State-Space/TransferFunction LTI, Evans Root Locus (`rlocus`), Butterworth/Welch signal filtering, and Primal-Dual Interior-Point `quadprog`/`linprog`. |
| **L4** | `plot`, `ai-local` | Dark-themed vector visualization (LTTB downsampling, SVG root locus, Bode/Nyquist, contours) and 100% offline local AI (SINDy equation discovery, sparse regression, no-std Rust transpiler). |
| **L5** | `engine` | Unified session command broker, multi-level snapshot undo/redo history, external provider bridging, and workspace persistence. |
| **L6** | `mcp`, `ui-egui` | Anthropic Model Context Protocol (MCP) server for autonomous AI agent pairing, and Egui desktop GUI surface. |
| **L7** | `matcraft-cli`, `matcraft`, `xtask` | Interactive REPL CLI with self-diagnostics doctor, native desktop application binary, and CI architecture linter/fuzzer. |

---

## ⚡ Quickstart

### Prerequisites
- [Rust](https://rustup.rs/) (stable, edition 2021)

### Build & Test
```bash
# Verify all workspace crates compile cleanly
cargo check --workspace

# Run full test suite across all 19 crates
cargo test --workspace

# Run strict clippy verification
cargo clippy --workspace
```

### Launch CLI REPL
```bash
cargo run -p matcraft-cli
```

### Run Architecture & Panic Token Audit
```bash
cargo run -p xtask -- all
```

---

## 🛠️ Key Capabilities

- **Mathematical Engine**: Dense & Sparse linear algebra, Arnoldi iteration (`eigs`), Francis double-shift QR eigenvalues (`eig`), Matrix exponential via Padé scaling & squaring (`expm`), Moore-Penrose pseudoinverse (`pinv`).
- **Optimization**: Primal-Dual Interior-Point convex quadratic programming (`quadprog`) and linear programming (`linprog`), Nelder-Mead simplex (`fminsearch`), and Levenberg-Marquardt with exact forward autodiff.
- **Control Systems**: State-space and transfer function representations, Kalman filters, model predictive control (MPC), and Evans Root Locus (`rlocus`) with complex s-plane pole/zero plotting.
- **Embedded Codegen**: Autonomous transpilation of continuous/discrete dynamic block diagrams and Stateflow charts into standalone, zero-allocation `#![no_std]` Rust and ANSI C.
- **Local Scientific AI**: SINDy (Sparse Identification of Nonlinear Dynamics) for discovering analytical ODEs directly from noisy time-series data without GPU or internet connectivity.
