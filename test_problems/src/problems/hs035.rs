use optimization::{SymbolicNlpOutputs, TypedRuntimeNlpBounds};
use sx_core::SX;

use super::{
    CaseMetadata, ProblemCase, VecN, exact_solution_validation, make_typed_case, symbolic_compile,
};

pub(crate) fn case() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), (), SX, _, _>(
        CaseMetadata::new(
            "hs035",
            "hock_schittkowski",
            "hs035",
            "manual",
            "HS035 with runtime lower bounds and one nonlinear inequality",
            false,
        ),
        |jit_opt_level| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), (), SX, _>(
                "hs035",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: 9.0 - 8.0 * x0 - 6.0 * x1 - 4.0 * x2
                            + 2.0 * x0.sqr()
                            + 2.0 * x1.sqr()
                            + x2.sqr()
                            + 2.0 * x0 * x1
                            + 2.0 * x0 * x2,
                        equalities: (),
                        inequalities: x0 + x1 + 2.0 * x2 - 3.0,
                    }
                },
                jit_opt_level,
            )?;
            Ok(super::TypedProblemData {
                compiled,
                x0: VecN {
                    values: [0.5, 0.5, 0.5],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [0.0, 0.0, 0.0],
                    }),
                    variable_upper: None,
                    inequality_lower: Some(-f64::INFINITY),
                    inequality_upper: Some(0.0),
                },
            })
        },
        exact_solution_validation(
            &[4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0],
            1e-4,
            1.0 / 9.0,
            1e-8,
            1e-5,
            1e-5,
            Some(1e-5),
        ),
    )
}
