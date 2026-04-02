# Testing

## Layers

### Unit tests

- symbolic graph invariants
- CCS invariants
- AD identities
- solver regressions

### Property tests

- randomized smooth operator checks
- randomized Hessian-strategy equivalence
- domain-restricted derivative coverage

### LLVM derivative tests

The LLVM JIT derivative suite checks compiled functions against:

- exact analytic derivatives
- finite-difference approximations
- the shared test-only symbolic evaluator

This combination catches:

- incorrect AD rules
- incorrect exact formulas in tests
- lowering / JIT backend mismatches

### CasADi parity

The parity audit distinguishes:

- exact supported
- intentionally de-scoped
- unsupported
- untracked

Generate the report with:

```bash
cargo run -p xtask -- casadi-parity-report
```

### Optimization tests

Optimization regressions run through compiled LLVM callbacks and cover:

- constrained Rosenbrock
- Hock-Schittkowski problems
- parameterized NLPs
- hanging chain

Both LLVM AOT and LLVM JIT backends are exercised.

There are also public-surface tests for:

- `#[derive(Vectorize)]` -> `symbolic_nlp(...)` -> `compile_jit()` -> `solve_sqp(...)`
- `#[derive(Vectorize)]` -> `symbolic_nlp(...)` -> `compile_jit()` -> `solve_interior_point(...)`
- `#[derive(Vectorize)]` -> `symbolic_nlp(...)` -> `compile_jit()` -> `solve_ipopt(...)`
- generated borrowed views projected from solver result buffers
- nested and const-generic layout roundtrips
- typed SQP iteration callbacks
- feature-gated serde roundtrips for public SQP transport DTOs
- feature-gated serde roundtrips for native IP and IPOPT transport DTOs
- honest optional metrics in snapshots and summaries
- strict non-finite input / callback rejection
