# Symbolic JIT NLP API

The public symbolic NLP path is now the typed path:

1. define symbolic variables and parameters from Rust types
2. build the symbolic objective / constraint expressions in one closure
3. JIT-compile the resulting NLP into reusable callbacks
4. solve it with runtime variable and constraint bounds

The old builder-style `SXMatrix` API and raw symbolic escape hatch are gone from the public API.

## Main entrypoints

- `#[derive(optimization::Vectorize)]`
- `optimization::symbolic_nlp(...)`
- `TypedSymbolicNlp::compile_jit()`
- `TypedCompiledJitNlp::solve_sqp(...)`
- `TypedCompiledJitNlp::solve_interior_point(...)`
- `TypedCompiledJitNlp::solve_ipopt(...)`
- `optimization::flat_view(...)`
- `solve_nlp_sqp_with_callback(...)`

## Why this shape

- dimensions come from the Rust type, not from user-supplied matrix sizes
- users define the NLP once as `(X, P) -> { objective, constraints }`
- gradients, Jacobians, and exact Lagrangian Hessians are derived automatically
- runtime variable bounds and nonlinear constraint bounds stay solver inputs, not compile-time state
- the same source type also defines flatten order, owned roundtrips, and generated borrowed numeric views

## Example

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
assert_eq!(*state.x, 1.0);
```

## Vectorize scope

Today the typed symbolic layer supports owned scalar-leaf schemas:

- scalar leaves `T`
- fixed arrays `[T; N]`
- nested structs whose fields are themselves vectorizable
- const-generic container structs like `Foo<T, const N: usize>`

The intended symbolic leaf type is `SX`, not `SXMatrix`.

That keeps the typed layer scalar-structured and avoids reintroducing dynamic per-field sizes into the API.

## Runtime bounds

Bounds are runtime-only:

- variable lower / upper bounds
- nonlinear constraint lower / upper bounds

Constraint semantics are reconstructed from the runtime lower/upper bounds:

- `lower == upper` => equality
- finite lower only => lower inequality
- finite upper only => upper inequality
- both finite and different => two inequalities

## Telemetry model

SQP exposes a typed callback surface:

- `solve_nlp_sqp_with_callback(...)`
- `SqpIterationSnapshot`
- `SqpIterationPhase`
- `SqpLineSearchInfo`
- `SqpQpInfo`
- `SqpIterationTiming`
- `SqpIterationEvent`
- `SqpTermination`

Important semantics:

- iteration `0` is the initial snapshot before the first accepted step
- unavailable metrics are `Option`, not fake `0` or `NaN`
- post-convergence recomputation is surfaced as an explicit `PostConvergence` snapshot
- reduced-accuracy QP solves are accepted and surfaced as typed QP status / event data

Console logging is just a renderer over the same public telemetry.

## Honest summary semantics

`ClarabelSqpSummary` carries:

- `termination`
- `final_state`
- `final_state_kind`
- `last_accepted_state`
- profiling and backend timing metadata

Constraint-family residuals remain explicit:

- `equality_inf_norm: Option<f64>`
- `inequality_inf_norm: Option<f64>`
- `complementarity_inf_norm: Option<f64>`

## Finite validation

SQP rejects non-finite values at the solver boundary:

- `x0`
- parameter values
- objective output
- objective gradient
- constraint values
- constraint Jacobian values
- Lagrangian Hessian values

Failures are typed:

- `ClarabelSqpError::NonFiniteInput`
- `ClarabelSqpError::NonFiniteCallbackOutput`

## Structured layout model

The typed layout layer now provides, from a single Rust source type:

- symbolic construction with `SX` leaves
- owned numeric flattening
- owned numeric unflattening
- generated borrowed numeric view types such as `AbView<'a, T>`
- borrowed flat-slice projection with `flat_view::<Ab<f64>, f64>(...)`

The supported scope is still intentionally narrow:

- scalar leaves only
- fixed arrays
- nested structs
- const-generic container structs

This keeps the symbolic/JIT NLP API typed, scalar-structured, and solver-focused without adding a broader schema/reflection system.

## Solver transport and serde

When the `optimization/serde` feature is enabled, the public SQP, native
interior-point, and IPOPT callback / summary DTOs derive `Serialize` /
`Deserialize`.

The transport contract is:

- timing fields stay `Duration` in Rust
- timing fields serialize as floating-point seconds
- aggregate callback timing stats serialize as `{ calls, total_time }`
- unavailable metrics remain explicit `Option`/typed variants, not placeholder
  `NaN` values

The public SQP diagnostics payloads are also transport-safe and owned:

- `SqpConeKind`
- `SqpQpRawStatus`
- `IpoptRawStatus`

so downstream apps do not need their own mirror enums for Clarabel statuses or
cone labels, and do not need to mirror upstream IPOPT status strings either.
