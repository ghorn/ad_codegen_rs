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
