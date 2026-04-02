# ad_codegen_rs

[![CI](https://github.com/ghorn/ad_codegen_rs/actions/workflows/ci.yml/badge.svg)](https://github.com/ghorn/ad_codegen_rs/actions/workflows/ci.yml)

`ad_codegen_rs` is a greenfield Rust implementation of the `SX` subset of CasADi, with:

- symbolic scalar expressions and sparse `SXMatrix`
- forward and reverse mode AD on `SXMatrix`
- canonical `CCS` sparsity
- LLVM JIT and LLVM AOT code generation
- exact-Hessian nonlinear optimization via SQP and IPOPT

This project is intentionally scoped around compiled execution. There is no public runtime interpreter for `SX`; evaluation happens through LLVM JIT or LLVM-generated native object code.

The current public optimization path is:

- define typed symbolic variables and parameters with `sx_core` + `optimization::Vectorize`
- build the NLP with `optimization::symbolic_nlp(...)`
- JIT-compile it with `.compile_jit()`
- solve it with runtime variable / constraint bounds
- project flat numeric buffers back into generated typed borrowed views with `optimization::flat_view(...)`

## Using From Another Project

These crates are intended to be consumed from Git for now. They are not published to crates.io yet, and the workspace currently marks packages `publish = false` to avoid accidental publication while the API and naming are still experimental.

Example:

```toml
[dependencies]
sx_core = { git = "https://github.com/ghorn/ad_codegen_rs.git" }
optimization = { git = "https://github.com/ghorn/ad_codegen_rs.git" }
```

With IPOPT enabled:

```toml
[dependencies]
optimization = { git = "https://github.com/ghorn/ad_codegen_rs.git", features = ["ipopt"] }
```

Cargo will pull the whole workspace and resolve the internal path dependencies automatically.

## Current Scope

Implemented today:

- `sx_core`
  - `SX`, `SXMatrix`, `CCS`, `SXFunction`
  - forward / reverse directional AD
  - `gradient`, `jacobian`, `hessian`
  - broad scalar math surface: `sin`, `cos`, `tan`, inverses, hyperbolics, `exp`, `log`, `pow`, `hypot`, `mod`, `abs`, `copysign`, `min`, `max`, and related helpers
- `sx_codegen`
  - backend-neutral lowering
- `sx_codegen_llvm`
  - LLVM JIT execution
  - LLVM AOT object emission plus Rust FFI/context wrapper generation
- `optimization`
  - exact-Hessian SQP on top of Clarabel
  - IPOPT backend behind the `ipopt` feature
  - support for equality constraints, nonlinear inequalities, and variable bounds
- `examples_source` / `examples_run`
  - Rosenbrock
  - constrained Rosenbrock
  - hanging chain
  - CasADi-compatible SX example tests where they fit the current scope

Not in scope yet:

- `MX`
- public runtime interpretation of `SX`
- comparison / conditional nodes in `SX`
- dense matrix representation as a first-class symbolic storage type

## Workspace Layout

- [sx_core](/Users/greg/dev/ad_codegen/sx_core): symbolic graph, sparse matrices, AD
- [sx_codegen](/Users/greg/dev/ad_codegen/sx_codegen): lowered IR
- [sx_codegen_llvm](/Users/greg/dev/ad_codegen/sx_codegen_llvm): LLVM JIT/AOT backend
- [optimization](/Users/greg/dev/ad_codegen/optimization): SQP and IPOPT adapters
- [examples_source](/Users/greg/dev/ad_codegen/examples_source): symbolic problem builders
- [examples_run](/Users/greg/dev/ad_codegen/examples_run): generated-code consumers and benches
- [bench_report](/Users/greg/dev/ad_codegen/bench_report): Markdown benchmark/report rendering
- [xtask](/Users/greg/dev/ad_codegen/xtask): report/audit runners

## Toolchain

- Rust `1.94`
- LLVM `22.1.x`
- macOS development currently assumes Homebrew LLVM:
  - `brew install llvm`

Optional:

- IPOPT tests and examples:
  - `brew install ipopt`

## Quick Start

### Base workspace

```bash
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

### Strict lint mode

```bash
cargo lint-strict
```

### IPOPT-enabled optimization tests

```bash
PKG_CONFIG_PATH="$(brew --prefix ipopt)/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
cargo test -p optimization --features ipopt
```

### Hanging-chain solve traces

SQP:

```bash
cargo test -p optimization --test sqp sqp_solves_hanging_chain -- --nocapture --test-threads=1
```

JIT backend specifically:

```bash
cargo test -p optimization --test sqp 'sqp_solves_hanging_chain::backend_2_CallbackBackend__Jit' -- --nocapture --test-threads=1
```

IPOPT:

```bash
PKG_CONFIG_PATH="$(brew --prefix ipopt)/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
cargo test -p optimization --features ipopt ipopt_solves_hanging_chain -- --nocapture
```

### Public symbolic JIT NLP path

```rust
use optimization::{
    ClarabelSqpOptions, SymbolicNlpOutputs, TypedRuntimeNlpBounds, flat_view, symbolic_nlp,
};
use sx_core::SX;

#[derive(Clone, optimization::Vectorize)]
struct Pair<T> {
    x: T,
    y: T,
}

let symbolic = symbolic_nlp::<Pair<SX>, (), (), _>("rosenbrock", |x, _| SymbolicNlpOutputs {
    objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
    equalities: (),
    inequalities: (),
})?;

let compiled = symbolic.compile_jit()?;
let summary = compiled.solve_sqp(
    &Pair { x: -1.2, y: 1.0 },
    &(),
    &TypedRuntimeNlpBounds::default(),
    &ClarabelSqpOptions::default(),
)?;

let state: PairView<'_, f64> = flat_view::<Pair<f64>, f64>(&summary.x)?;
```

## Reports

AD cost report:

```bash
cargo run -p xtask -- ad-cost-report
```

Parity audit report:

```bash
cargo run -p xtask -- casadi-parity-report
```

Test-problem suite:

```bash
cargo run --release -p test_problems -- --problem-set fast --solver both --jobs 4 --output-dir target/test-problems
```

Outputs are written under [target/reports](/Users/greg/dev/ad_codegen/target/reports).

## Testing Strategy

- unit tests for graph construction, CCS invariants, AD identities, and solver behavior
- property tests over safe numeric domains
- LLVM JIT numerical derivative checks against:
  - exact analytic formulas
  - finite differences
  - a shared test-only symbolic evaluator
- CasADi parity audit and exact test ports for the currently supported SX/CCS/AD surface

The symbolic evaluator used in tests is intentionally test-only. Public/runtime evaluation still goes through LLVM.

## Documentation

- [docs/getting-started.md](/Users/greg/dev/ad_codegen/docs/getting-started.md)
- [docs/architecture.md](/Users/greg/dev/ad_codegen/docs/architecture.md)
- [docs/symbolic-nlp.md](/Users/greg/dev/ad_codegen/docs/symbolic-nlp.md)
- [docs/upstream-followup.md](/Users/greg/dev/ad_codegen/docs/upstream-followup.md)
- [docs/testing.md](/Users/greg/dev/ad_codegen/docs/testing.md)

## License

Licensed under either:

- Apache License, Version 2.0, see [LICENSE-APACHE](/Users/greg/dev/ad_codegen/LICENSE-APACHE)
- MIT license, see [LICENSE-MIT](/Users/greg/dev/ad_codegen/LICENSE-MIT)

at your option.
