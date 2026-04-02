use optimization::{SymbolicNlpOutputs, TypedRuntimeNlpBounds, Vectorize};
use sx_core::SX;

use crate::model::{ProblemRunRecord, ValidationOutcome, ValidationTier};

use super::{CaseMetadata, Chain, Point, ProblemCase, VecN, make_typed_case, symbolic_compile};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConstraintMode {
    Equality,
    Inequality,
}

impl ConstraintMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Equality => "equality",
            Self::Inequality => "inequality",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InitialCondition {
    ZigZagFeasible,
    StraightAcross,
    QuadraticBestEffort,
}

pub(crate) fn cases() -> Vec<ProblemCase> {
    vec![
        case_for::<5, 6>(
            "hanging_chain_links06_eq_zigzag",
            "links=6, init=zigzag_feasible, constraints=equality",
            InitialCondition::ZigZagFeasible,
            ConstraintMode::Equality,
        ),
        case_for::<5, 6>(
            "hanging_chain_links06_eq_straight",
            "links=6, init=straight_infeasible, constraints=equality",
            InitialCondition::StraightAcross,
            ConstraintMode::Equality,
        ),
        case_for::<5, 6>(
            "hanging_chain_links06_eq_quadratic",
            "links=6, init=quadratic_best_effort, constraints=equality",
            InitialCondition::QuadraticBestEffort,
            ConstraintMode::Equality,
        ),
        case_for::<5, 6>(
            "hanging_chain_links06_ineq_zigzag",
            "links=6, init=zigzag_feasible, constraints=inequality",
            InitialCondition::ZigZagFeasible,
            ConstraintMode::Inequality,
        ),
        case_for::<5, 6>(
            "hanging_chain_links06_ineq_straight",
            "links=6, init=straight_infeasible, constraints=inequality",
            InitialCondition::StraightAcross,
            ConstraintMode::Inequality,
        ),
        case_for::<5, 6>(
            "hanging_chain_links06_ineq_quadratic",
            "links=6, init=quadratic_best_effort, constraints=inequality",
            InitialCondition::QuadraticBestEffort,
            ConstraintMode::Inequality,
        ),
        case_for::<11, 12>(
            "hanging_chain_links12_eq_zigzag",
            "links=12, init=zigzag_feasible, constraints=equality",
            InitialCondition::ZigZagFeasible,
            ConstraintMode::Equality,
        ),
        case_for::<11, 12>(
            "hanging_chain_links12_eq_straight",
            "links=12, init=straight_infeasible, constraints=equality",
            InitialCondition::StraightAcross,
            ConstraintMode::Equality,
        ),
        case_for::<11, 12>(
            "hanging_chain_links12_eq_quadratic",
            "links=12, init=quadratic_best_effort, constraints=equality",
            InitialCondition::QuadraticBestEffort,
            ConstraintMode::Equality,
        ),
        case_for::<11, 12>(
            "hanging_chain_links12_ineq_zigzag",
            "links=12, init=zigzag_feasible, constraints=inequality",
            InitialCondition::ZigZagFeasible,
            ConstraintMode::Inequality,
        ),
        case_for::<11, 12>(
            "hanging_chain_links12_ineq_straight",
            "links=12, init=straight_infeasible, constraints=inequality",
            InitialCondition::StraightAcross,
            ConstraintMode::Inequality,
        ),
        case_for::<11, 12>(
            "hanging_chain_links12_ineq_quadratic",
            "links=12, init=quadratic_best_effort, constraints=inequality",
            InitialCondition::QuadraticBestEffort,
            ConstraintMode::Inequality,
        ),
        case_for::<23, 24>(
            "hanging_chain_links24_eq_zigzag",
            "links=24, init=zigzag_feasible, constraints=equality",
            InitialCondition::ZigZagFeasible,
            ConstraintMode::Equality,
        ),
        case_for::<23, 24>(
            "hanging_chain_links24_eq_straight",
            "links=24, init=straight_infeasible, constraints=equality",
            InitialCondition::StraightAcross,
            ConstraintMode::Equality,
        ),
        case_for::<23, 24>(
            "hanging_chain_links24_eq_quadratic",
            "links=24, init=quadratic_best_effort, constraints=equality",
            InitialCondition::QuadraticBestEffort,
            ConstraintMode::Equality,
        ),
        case_for::<23, 24>(
            "hanging_chain_links24_ineq_zigzag",
            "links=24, init=zigzag_feasible, constraints=inequality",
            InitialCondition::ZigZagFeasible,
            ConstraintMode::Inequality,
        ),
        case_for::<23, 24>(
            "hanging_chain_links24_ineq_straight",
            "links=24, init=straight_infeasible, constraints=inequality",
            InitialCondition::StraightAcross,
            ConstraintMode::Inequality,
        ),
        case_for::<23, 24>(
            "hanging_chain_links24_ineq_quadratic",
            "links=24, init=quadratic_best_effort, constraints=inequality",
            InitialCondition::QuadraticBestEffort,
            ConstraintMode::Inequality,
        ),
        case_for::<47, 48>(
            "hanging_chain_links48_eq_zigzag",
            "links=48, init=zigzag_feasible, constraints=equality",
            InitialCondition::ZigZagFeasible,
            ConstraintMode::Equality,
        ),
        case_for::<47, 48>(
            "hanging_chain_links48_eq_straight",
            "links=48, init=straight_infeasible, constraints=equality",
            InitialCondition::StraightAcross,
            ConstraintMode::Equality,
        ),
        case_for::<47, 48>(
            "hanging_chain_links48_eq_quadratic",
            "links=48, init=quadratic_best_effort, constraints=equality",
            InitialCondition::QuadraticBestEffort,
            ConstraintMode::Equality,
        ),
        case_for::<47, 48>(
            "hanging_chain_links48_ineq_zigzag",
            "links=48, init=zigzag_feasible, constraints=inequality",
            InitialCondition::ZigZagFeasible,
            ConstraintMode::Inequality,
        ),
        case_for::<47, 48>(
            "hanging_chain_links48_ineq_straight",
            "links=48, init=straight_infeasible, constraints=inequality",
            InitialCondition::StraightAcross,
            ConstraintMode::Inequality,
        ),
        case_for::<47, 48>(
            "hanging_chain_links48_ineq_quadratic",
            "links=48, init=quadratic_best_effort, constraints=inequality",
            InitialCondition::QuadraticBestEffort,
            ConstraintMode::Inequality,
        ),
    ]
}

fn case_for<const N: usize, const LINKS: usize>(
    id: &'static str,
    variant: &'static str,
    initial_condition: InitialCondition,
    constraint_mode: ConstraintMode,
) -> ProblemCase {
    let metadata = CaseMetadata::new(
        id,
        "hanging_chain",
        variant,
        "manual",
        "Parameterized hanging chain with runtime equality/inequality link bounds and configurable initial condition",
        true,
    );
    match constraint_mode {
        ConstraintMode::Equality => make_typed_case::<Chain<SX, N>, (), VecN<SX, LINKS>, (), _, _>(
            metadata,
            move |jit_opt_level| {
                let spec = HangingChainSpec::for_links(LINKS);
                let compiled = symbolic_compile::<Chain<SX, N>, (), VecN<SX, LINKS>, (), _>(
                    id,
                    |q, ()| {
                        let objective = hanging_chain_objective(q);
                        let constraints = hanging_chain_constraints::<N, LINKS>(q, spec);
                        SymbolicNlpOutputs {
                            objective,
                            equalities: constraints,
                            inequalities: (),
                        }
                    },
                    jit_opt_level,
                )?;
                Ok(super::TypedProblemData {
                    compiled,
                    x0: initial_guess::<N, LINKS>(spec, initial_condition),
                    parameters: (),
                    bounds: TypedRuntimeNlpBounds::default(),
                })
            },
            move |record| validate_hanging_chain::<N, LINKS>(record, constraint_mode),
        ),
        ConstraintMode::Inequality => {
            make_typed_case::<Chain<SX, N>, (), (), VecN<SX, LINKS>, _, _>(
                metadata,
                move |jit_opt_level| {
                    let spec = HangingChainSpec::for_links(LINKS);
                    let compiled = symbolic_compile::<Chain<SX, N>, (), (), VecN<SX, LINKS>, _>(
                        id,
                        |q, ()| {
                            let objective = hanging_chain_objective(q);
                            let constraints = hanging_chain_constraints::<N, LINKS>(q, spec);
                            SymbolicNlpOutputs {
                                objective,
                                equalities: (),
                                inequalities: constraints,
                            }
                        },
                        jit_opt_level,
                    )?;
                    Ok(super::TypedProblemData {
                        compiled,
                        x0: initial_guess::<N, LINKS>(spec, initial_condition),
                        parameters: (),
                        bounds: TypedRuntimeNlpBounds {
                            variable_lower: None,
                            variable_upper: None,
                            inequality_lower: Some(
                                <VecN<SX, LINKS> as Vectorize<SX>>::from_flat_fn(&mut || {
                                    f64::NEG_INFINITY
                                }),
                            ),
                            inequality_upper: Some(
                                <VecN<SX, LINKS> as Vectorize<SX>>::from_flat_fn(&mut || 0.0),
                            ),
                        },
                    })
                },
                move |record| validate_hanging_chain::<N, LINKS>(record, constraint_mode),
            )
        }
    }
}

#[derive(Clone, Copy)]
struct HangingChainSpec {
    anchor_left: (f64, f64),
    anchor_right: (f64, f64),
    link_length: f64,
    link_length_sq: f64,
}

impl HangingChainSpec {
    fn for_links(links: usize) -> Self {
        let link_length = 1.0;
        let span = 0.75 * links as f64;
        Self {
            anchor_left: (0.0, 0.0),
            anchor_right: (span, 0.0),
            link_length,
            link_length_sq: link_length * link_length,
        }
    }
}

fn hanging_chain_objective<const N: usize>(q: &Chain<SX, N>) -> SX {
    q.points.iter().fold(SX::zero(), |acc, point| acc + point.y)
}

fn hanging_chain_constraints<const N: usize, const LINKS: usize>(
    q: &Chain<SX, N>,
    spec: HangingChainSpec,
) -> VecN<SX, LINKS> {
    VecN {
        values: std::array::from_fn(|idx| {
            distance_constraint::<N>(
                q,
                idx,
                spec.anchor_left,
                spec.anchor_right,
                spec.link_length_sq,
            )
        }),
    }
}

fn distance_constraint<const N: usize>(
    q: &Chain<SX, N>,
    idx: usize,
    anchor_left: (f64, f64),
    anchor_right: (f64, f64),
    link_length_sq: f64,
) -> SX {
    if idx == 0 {
        let point = &q.points[0];
        (point.x - anchor_left.0).sqr() + (point.y - anchor_left.1).sqr() - link_length_sq
    } else if idx == N {
        let point = &q.points[N - 1];
        (point.x - anchor_right.0).sqr() + (point.y - anchor_right.1).sqr() - link_length_sq
    } else {
        let left = &q.points[idx - 1];
        let right = &q.points[idx];
        (right.x - left.x).sqr() + (right.y - left.y).sqr() - link_length_sq
    }
}

fn initial_guess<const N: usize, const LINKS: usize>(
    spec: HangingChainSpec,
    initial_condition: InitialCondition,
) -> Chain<f64, N> {
    match initial_condition {
        InitialCondition::ZigZagFeasible => zigzag_feasible_guess::<N, LINKS>(spec),
        InitialCondition::StraightAcross => straight_across_guess::<N, LINKS>(spec),
        InitialCondition::QuadraticBestEffort => quadratic_best_effort_guess::<N, LINKS>(spec),
    }
}

fn zigzag_feasible_guess<const N: usize, const LINKS: usize>(
    spec: HangingChainSpec,
) -> Chain<f64, N> {
    let shape: [f64; N] = std::array::from_fn(|idx| if idx % 2 == 0 { -1.0 } else { 1.0 });
    let mut dy_coeffs = [0.0; LINKS];
    for idx in 0..LINKS {
        let current = if idx < N { shape[idx] } else { 0.0 };
        let previous = if idx == 0 { 0.0 } else { shape[idx - 1] };
        dy_coeffs[idx] = current - previous;
    }
    let max_dy_coeff = dy_coeffs
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let max_amplitude = if max_dy_coeff > 0.0 {
        0.999 * spec.link_length / max_dy_coeff
    } else {
        0.0
    };
    let target_span = spec.anchor_right.0 - spec.anchor_left.0;
    let mut low = 0.0;
    let mut high = max_amplitude;
    for _ in 0..96 {
        let mid = 0.5 * (low + high);
        if horizontal_span(mid, &dy_coeffs, spec.link_length) > target_span {
            low = mid;
        } else {
            high = mid;
        }
    }
    let amplitude = high;
    let y_values = std::array::from_fn(|idx| amplitude * shape[idx]);
    chain_from_sequential_y(spec, &y_values)
}

fn straight_across_guess<const N: usize, const LINKS: usize>(
    spec: HangingChainSpec,
) -> Chain<f64, N> {
    let span = spec.anchor_right.0 - spec.anchor_left.0;
    let points = std::array::from_fn(|idx| {
        let t = (idx + 1) as f64 / LINKS as f64;
        Point {
            x: spec.anchor_left.0 + span * t,
            y: 0.0,
        }
    });
    Chain { points }
}

fn quadratic_best_effort_guess<const N: usize, const LINKS: usize>(
    spec: HangingChainSpec,
) -> Chain<f64, N> {
    let span = spec.anchor_right.0 - spec.anchor_left.0;
    let dx = span / LINKS as f64;
    let max_amplitude = max_quadratic_amplitude::<N>(dx, spec.link_length);
    let amplitude = 0.95 * max_amplitude;
    let points = std::array::from_fn(|idx| {
        let t = (idx + 1) as f64 / LINKS as f64;
        let shape = 4.0 * t * (1.0 - t);
        Point {
            x: spec.anchor_left.0 + span * t,
            y: -amplitude * shape,
        }
    });
    Chain { points }
}

fn max_quadratic_amplitude<const N: usize>(dx: f64, link_length: f64) -> f64 {
    let mut low = 0.0;
    let mut high = link_length;
    for _ in 0..96 {
        let mid = 0.5 * (low + high);
        if quadratic_guess_is_feasible::<N>(dx, link_length, mid) {
            low = mid;
        } else {
            high = mid;
        }
    }
    low
}

fn quadratic_guess_is_feasible<const N: usize>(dx: f64, link_length: f64, amplitude: f64) -> bool {
    let y_values: [f64; N] = std::array::from_fn(|idx| {
        let t = (idx + 1) as f64 / (N + 1) as f64;
        -amplitude * 4.0 * t * (1.0 - t)
    });
    let mut previous_y = 0.0;
    for current_y in y_values.into_iter().chain(std::iter::once(0.0)) {
        let dy = current_y - previous_y;
        if dx * dx + dy * dy > link_length * link_length {
            return false;
        }
        previous_y = current_y;
    }
    true
}

fn chain_from_sequential_y<const N: usize>(
    spec: HangingChainSpec,
    y_values: &[f64; N],
) -> Chain<f64, N> {
    let mut x_cursor = spec.anchor_left.0;
    let points = std::array::from_fn(|idx| {
        let previous_y = if idx == 0 {
            spec.anchor_left.1
        } else {
            y_values[idx - 1]
        };
        let dy = y_values[idx] - previous_y;
        let dx = (spec.link_length_sq - dy * dy).sqrt();
        x_cursor += dx;
        Point {
            x: x_cursor,
            y: y_values[idx],
        }
    });
    Chain { points }
}

fn horizontal_span(amplitude: f64, dy_coeffs: &[f64], link: f64) -> f64 {
    dy_coeffs.iter().fold(0.0, |acc, coeff| {
        let dy = amplitude * *coeff;
        acc + (link * link - dy * dy).sqrt()
    })
}

fn validate_hanging_chain<const N: usize, const LINKS: usize>(
    record: &ProblemRunRecord,
    mode: ConstraintMode,
) -> ValidationOutcome {
    let Some(solution) = record.solution.as_ref() else {
        return ValidationOutcome {
            tier: ValidationTier::Failed,
            tolerance: tolerance_text(mode, LINKS),
            detail: "missing solution".to_string(),
        };
    };
    let primal = record.metrics.primal_inf.unwrap_or(f64::INFINITY);
    let dual = record.metrics.dual_inf.unwrap_or(f64::INFINITY);
    let objective = record.metrics.objective.unwrap_or(f64::INFINITY);
    let comp = record.metrics.complementarity_inf;
    let span = 0.75 * LINKS as f64;
    let mut symmetry_error = 0.0_f64;
    for idx in 0..N {
        let mirrored = N - 1 - idx;
        let xi = solution[2 * idx];
        let yi = solution[2 * idx + 1];
        let xj = solution[2 * mirrored];
        let yj = solution[2 * mirrored + 1];
        symmetry_error = symmetry_error.max(((xi + xj) - span).abs());
        symmetry_error = symmetry_error.max((yi - yj).abs());
    }

    let primal_tol = if LINKS >= 48 { 5e-5 } else { 1e-5 };
    let dual_tol = if LINKS >= 24 { 1e-3 } else { 5e-4 };
    let symmetry_tol = if LINKS >= 48 {
        8e-2
    } else if LINKS >= 24 {
        5e-2
    } else {
        2e-2
    };
    let comp_tol = if matches!(mode, ConstraintMode::Inequality) {
        Some(if LINKS >= 24 { 5e-4 } else { 1e-4 })
    } else {
        None
    };
    let mut passed = primal <= primal_tol
        && dual <= dual_tol
        && objective.is_finite()
        && objective < -1e-3
        && symmetry_error <= symmetry_tol;
    if let (Some(limit), Some(comp_value)) = (comp_tol, comp) {
        passed &= comp_value <= limit;
    }
    let tier = if passed {
        ValidationTier::Passed
    } else if record.error.is_none()
        && primal <= 1e-6
        && dual <= 1e-6
        && comp.is_none_or(|value| value <= 1e-6)
    {
        ValidationTier::ReducedAccuracy
    } else {
        ValidationTier::Failed
    };
    ValidationOutcome {
        tier,
        tolerance: tolerance_text(mode, LINKS),
        detail: format!(
            "objective={objective:.6e}, primal={primal:.3e}, dual={dual:.3e}, comp={}, symmetry={symmetry_error:.3e}, mode={}",
            comp.map_or_else(|| "--".to_string(), |value| format!("{value:.3e}")),
            mode.as_str(),
        ),
    }
}

fn tolerance_text(mode: ConstraintMode, links: usize) -> String {
    let primal_tol = if links >= 48 { 5e-5 } else { 1e-5 };
    let dual_tol = if links >= 24 { 1e-3 } else { 5e-4 };
    let symmetry_tol = if links >= 48 {
        8e-2
    } else if links >= 24 {
        5e-2
    } else {
        2e-2
    };
    match mode {
        ConstraintMode::Equality => {
            format!(
                "objective<0, primal<={primal_tol:.1e}, dual<={dual_tol:.1e}, symmetry<={symmetry_tol:.1e}"
            )
        }
        ConstraintMode::Inequality => {
            let comp_tol = if links >= 24 { 5e-4 } else { 1e-4 };
            format!(
                "objective<0, primal<={primal_tol:.1e}, dual<={dual_tol:.1e}, comp<={comp_tol:.1e}, symmetry<={symmetry_tol:.1e}"
            )
        }
    }
}
