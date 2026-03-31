# Getting Started

## Prerequisites

- Rust `1.94`
- LLVM `22.1.x`
- macOS with Homebrew is the currently exercised developer setup

Install LLVM:

```bash
brew install llvm
```

Optional IPOPT support:

```bash
brew install ipopt
```

## Common Commands

Base workspace checks:

```bash
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Strict lint mode:

```bash
cargo lint-strict
```

Coverage summary:

```bash
cargo llvm-cov --workspace --summary-only
```

## Optimization Runs

Public symbolic JIT NLP tests:

```bash
cargo test -p optimization --test symbolic_jit_nlp
```

SQP callback / finite-validation tests:

```bash
cargo test -p optimization --test sqp_callback_api
```

SQP hanging chain:

```bash
cargo test -p optimization --test sqp sqp_solves_hanging_chain -- --nocapture --test-threads=1
```

IPOPT hanging chain:

```bash
PKG_CONFIG_PATH="$(brew --prefix ipopt)/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
cargo test -p optimization --features ipopt ipopt_solves_hanging_chain -- --nocapture
```

## Reports

AD cost:

```bash
cargo run -p xtask -- ad-cost-report
```

CasADi parity:

```bash
cargo run -p xtask -- casadi-parity-report
```
