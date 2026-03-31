use optimization::{SymbolicNlpOutputs, TypedRuntimeNlpBounds};
use sx_core::SX;

use super::{
    CaseMetadata, ProblemCase, VecN, exact_solution_validation, make_typed_case, symbolic_compile,
};

pub(crate) fn case() -> ProblemCase {
    make_typed_case::<VecN<SX, 4>, (), (), _, _>(
        CaseMetadata::new("wood_4", "wood", "n=4", "manual", "Wood function", false),
        |jit_opt_level| {
            let compiled = symbolic_compile::<VecN<SX, 4>, (), (), _>(
                "wood_4",
                |x, ()| {
                    let x1 = x.values[0];
                    let x2 = x.values[1];
                    let x3 = x.values[2];
                    let x4 = x.values[3];
                    let objective = 100.0 * (x1.sqr() - x2).sqr()
                        + (x1 - 1.0).sqr()
                        + (x3 - 1.0).sqr()
                        + 90.0 * (x3.sqr() - x4).sqr()
                        + 10.1 * ((x2 - 1.0).sqr() + (x4 - 1.0).sqr())
                        + 19.8 * (x2 - 1.0) * (x4 - 1.0);
                    SymbolicNlpOutputs {
                        objective,
                        constraints: (),
                    }
                },
                jit_opt_level,
            )?;
            Ok(super::TypedProblemData {
                compiled,
                x0: VecN {
                    values: [-3.0, -1.0, -3.0, -1.0],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        exact_solution_validation(&[1.0, 1.0, 1.0, 1.0], 2e-4, 0.0, 1e-8, 1e-6, 1e-5, None),
    )
}
