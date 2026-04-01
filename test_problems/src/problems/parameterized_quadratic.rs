use optimization::{SymbolicNlpOutputs, TypedRuntimeNlpBounds};
use sx_core::SX;

use super::{
    CaseMetadata, Pair, ProblemCase, exact_solution_validation, make_typed_case, symbolic_compile,
};

pub(crate) fn case() -> ProblemCase {
    make_typed_case::<Pair<SX>, Pair<SX>, SX, (), _, _>(
        CaseMetadata::new(
            "parameterized_quadratic",
            "parameterized_quadratic",
            "sum(x)=1",
            "manual",
            "Parameterized quadratic with runtime equality bounds",
            true,
        ),
        |jit_opt_level| {
            let compiled = symbolic_compile::<Pair<SX>, Pair<SX>, SX, (), _>(
                "parameterized_quadratic",
                |x, p| SymbolicNlpOutputs {
                    objective: (x.x - p.x).sqr() + (x.y - p.y).sqr(),
                    equalities: x.x + x.y,
                    inequalities: (),
                },
                jit_opt_level,
            )?;
            Ok(super::TypedProblemData {
                compiled,
                x0: Pair { x: 0.9, y: 0.1 },
                parameters: Pair { x: 0.25, y: 0.75 },
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: None,
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: None,
                },
            })
        },
        exact_solution_validation(&[0.25, 0.75], 1e-6, 0.0, 1e-9, 1e-8, 1e-6, None),
    )
}
