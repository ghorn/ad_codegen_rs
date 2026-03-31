use optimization::{SymbolicNlpOutputs, TypedRuntimeNlpBounds};
use sx_core::SX;

use super::{
    CaseMetadata, Pair, ProblemCase, exact_solution_validation, make_typed_case, symbolic_compile,
};

pub(crate) fn case() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), SX, _, _>(
        CaseMetadata::new(
            "hs021",
            "hock_schittkowski",
            "hs021",
            "manual",
            "HS021 with runtime box bounds and one nonlinear inequality",
            false,
        ),
        |jit_opt_level| {
            let compiled = symbolic_compile::<Pair<SX>, (), SX, _>(
                "hs021",
                |x, ()| SymbolicNlpOutputs {
                    objective: 0.01 * x.x.sqr() + x.y.sqr() - 100.0,
                    constraints: -10.0 * x.x + x.y + 10.0,
                },
                jit_opt_level,
            )?;
            Ok(super::TypedProblemData {
                compiled,
                x0: Pair { x: 2.0, y: 2.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair { x: 2.0, y: -50.0 }),
                    variable_upper: Some(Pair { x: 50.0, y: 50.0 }),
                    constraint_lower: Some(-f64::INFINITY),
                    constraint_upper: Some(0.0),
                },
            })
        },
        exact_solution_validation(&[2.0, 0.0], 1e-5, -99.96, 1e-8, 1e-6, 1e-5, Some(1e-5)),
    )
}
