# Upstream Follow-up Status

This note records which parts of the recent downstream follow-up request were already present, which were missing, and what the current public contract is.

## Already present before the follow-up pass

- typed symbolic NLP construction through `symbolic_nlp(...)`
- typed `SX`-leaf layout derivation with `#[derive(Vectorize)]`
- runtime variable and nonlinear constraint bounds
- typed SQP callback snapshots and typed termination/failure enums
- explicit optional metrics in SQP snapshots/summaries
- fail-fast non-finite validation at SQP solver boundaries

## Finished in the follow-up pass

- generated borrowed numeric view types from the same source layout type
- explicit flatten/unflatten roundtrip coverage for nested and const-generic layouts
- explicit end-to-end examples/tests that project solver outputs back into typed views
- public docs that describe the typed layout / borrowed view model instead of calling it deferred
- optional `optimization/serde` support on the public SQP callback / summary / diagnostics DTOs
- transport-safe owned SQP diagnostics enums instead of downstream Clarabel/string mirror types
- explicit adapter-side SQP timing buckets for typed symbolic/JIT problems

## High-level DSL design

The intended ergonomic symbolic API is:

1. define a Rust source type such as `State<T>`
2. derive `Vectorize`
3. build a symbolic NLP by writing one closure from typed symbolic variables/parameters to:
   - `objective`
   - `equalities`
   - `inequalities`
4. compile it once with JIT
5. solve it with runtime bounds

This avoids hand-written callback glue while keeping the solver-facing runtime representation flat and efficient.

## Typed layout and generated borrowed views

The Rust type is the schema.

From one source type, the public API provides:

- symbolic construction
- owned numeric flatten
- owned numeric unflatten
- generated borrowed view types such as `FooView<'a, T>`
- borrowed flat-slice projection with `flat_view::<Foo<f64>, f64>(...)`

Field order defines flatten order. There is no string-based field lookup and no duplicated source-of-truth type.

## Unavailable metrics

Unavailable solver metrics are represented explicitly with typed optionality:

- iteration callbacks use `Option<f64>` / `Option<...>`
- final summaries use the same rule

The public contract does not encode “missing” or “not computed” as `NaN`, `Inf`, or placeholder floats.

## SQP transport and telemetry

The remaining downstream mirror structs this pass removes are:

- QP failure diagnostics DTOs
- line-search / rejected-trial DTOs
- cone summary DTOs
- per-iteration and aggregate timing wrappers
- local timing wrappers used only to split callback evaluation from adapter work

The public SQP transport-facing types can now be serialized behind the optional
`optimization/serde` feature.

Timing fields remain `Duration` in the Rust API, but serialize as floating-point
seconds under `serde`. Aggregate callback timing stats serialize as:

- `calls`
- `total_time`

The public QP-facing payloads are transport-safe and owned:

- `SqpConeKind::{Zero, Nonnegative, Other(String)}`
- `SqpQpRawStatus::{Solved, AlmostSolved, ..., Other(String)}`

That means downstream apps no longer need to mirror `clarabel::SolverStatus`
or string cone kinds just to move SQP telemetry across process/UI boundaries.

Adapter-side timing is also explicit:

- per-iteration `SqpIterationTiming::adapter_timing: Option<SqpAdapterTiming>`
- aggregate `ClarabelSqpProfiling::adapter_timing: Option<SqpAdapterTiming>`

The split is:

- user callback / compiled-kernel evaluation
- adapter output marshalling
- layout projection / flatten-view work

Handwritten `CompiledNlpProblem` implementations report these buckets as
unavailable rather than fake zeroes.

## Native IP / IPOPT transport parity

The same transport-facing principles now also apply to the native interior-point
solver and the IPOPT adapter.

Native interior-point now exposes:

- `InteriorPointIterationSnapshot`
- `InteriorPointIterationTiming`
- `InteriorPointProfiling`
- `InteriorPointSummary`
- `InteriorPointSolveError`

IPOPT now exposes:

- `IpoptIterationSnapshot`
- `IpoptIterationTiming`
- `IpoptProfiling`
- `IpoptSummary`
- `IpoptSolveError`
- `IpoptRawStatus`
- `solve_nlp_ipopt_with_callback(...)`

For both backends:

- public DTOs are `serde`-serializable behind `optimization/serde`
- timing stays `Duration` in Rust and serializes as floating-point seconds
- typed symbolic/JIT problems expose adapter timing explicitly
- handwritten `CompiledNlpProblem` implementations report adapter timing as unavailable

The IPOPT-facing transport contract is intentionally “as much parity as
reasonable without patching IPOPT itself”:

- public status payloads are owned and transport-safe via `IpoptRawStatus`
- per-iteration snapshots are available through the public callback entrypoint
- aggregate profiling is public
- journal / iteration richness is still bounded by what the upstream IPOPT Rust
  binding exposes without carrying a local fork

## Finite-value validation

SQP rejects non-finite values at solver boundaries:

- initial guess
- parameter values
- objective output
- objective gradient
- equality values
- inequality values
- equality Jacobian values
- inequality Jacobian values
- Lagrangian Hessian values

Failures are returned as typed errors naming the stage. Non-finite values are not forwarded into Clarabel and are not smuggled into telemetry as fake numbers.

## Tradeoffs

- The typed layout system is intentionally limited to scalar leaves, fixed arrays, nested structs, and const-generic container structs.
- This keeps the API predictable and easy to derive, but it is not a general reflection/schema framework.
- The symbolic NLP DSL stays typed and ergonomic without introducing a second, stringly schema system.
- Runtime-sized repeated typed blocks remain intentionally deferred. Fixed-layout generated borrowed views are present today; runtime-sized repeated layouts are the missing piece.
