use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config, TestRunner};
use sx_codegen_llvm::{CompiledJitFunction, LlvmOptimizationLevel};
use sx_core::{NamedMatrix, SX, SXFunction, SXMatrix};

#[path = "../../test_support/symbolic_eval.rs"]
mod symbolic_eval;

use symbolic_eval::eval;

const SAMPLE_COUNT: usize = 48;
const GRAD_STEP: f64 = 1e-6;
const HESS_STEP: f64 = 1e-4;

type UnaryBuilder = fn(SX) -> SX;
type UnaryReference = fn(f64) -> f64;
type UnarySampler = fn(&mut TestRunner) -> f64;
type BinaryBuilder = fn(SX, SX) -> SX;
type BinaryReference = fn(f64, f64) -> f64;
type BinarySampler = fn(&mut TestRunner) -> (f64, f64);

struct UnaryCase {
    label: &'static str,
    builder: UnaryBuilder,
    reference: UnaryReference,
    sampler: UnarySampler,
    grad_abs_tol: f64,
    grad_rel_tol: f64,
    hess_abs_tol: Option<f64>,
    hess_rel_tol: Option<f64>,
}

struct BinaryCase {
    label: &'static str,
    builder: BinaryBuilder,
    reference: BinaryReference,
    sampler: BinarySampler,
    grad_abs_tol: f64,
    grad_rel_tol: f64,
}

struct UnaryExactCase {
    label: &'static str,
    builder: UnaryBuilder,
    point: f64,
    value: UnaryReference,
    gradient: UnaryReference,
    hessian: Option<UnaryReference>,
    grad_abs_tol: f64,
    grad_rel_tol: f64,
    hess_abs_tol: Option<f64>,
    hess_rel_tol: Option<f64>,
}

struct BinaryExactCase {
    label: &'static str,
    builder: BinaryBuilder,
    point: (f64, f64),
    value: BinaryReference,
    grad_x: BinaryReference,
    grad_y: BinaryReference,
    grad_abs_tol: f64,
    grad_rel_tol: f64,
}

struct UnaryBundle {
    compiled: CompiledJitFunction,
    symbolic: SXFunction,
}

struct BinaryBundle {
    compiled: CompiledJitFunction,
    symbolic: SXFunction,
}

impl UnaryBundle {
    fn compile(name: &str, builder: UnaryBuilder, include_hessian: bool) -> Self {
        let x = SX::sym(format!("{name}_x"));
        let wrt = SXMatrix::dense_column(vec![x]).unwrap();
        let value = SXMatrix::scalar(builder(x));
        let gradient = value.gradient(&wrt).unwrap();
        let hessian = include_hessian.then(|| value.hessian(&wrt).unwrap());
        let mut outputs = vec![
            NamedMatrix::new("value", value).unwrap(),
            NamedMatrix::new("gradient", gradient).unwrap(),
        ];
        if let Some(hessian) = hessian {
            outputs.push(NamedMatrix::new("hessian", hessian).unwrap());
        }
        let function =
            SXFunction::new(name, vec![NamedMatrix::new("x", wrt).unwrap()], outputs).unwrap();
        let compiled =
            CompiledJitFunction::compile_function(&function, LlvmOptimizationLevel::O2).unwrap();
        Self {
            compiled,
            symbolic: function,
        }
    }

    fn eval(&self, x: f64) -> (f64, f64, Option<f64>) {
        let mut context = self.compiled.create_context();
        context.input_mut(0)[0] = x;
        self.compiled.eval(&mut context);
        let value = context.output(0)[0];
        let gradient = context.output(1)[0];
        let hessian = if self.compiled.lowered().outputs.len() > 2 {
            Some(context.output(2)[0])
        } else {
            None
        };
        (value, gradient, hessian)
    }

    fn eval_symbolic(&self, x: f64) -> Vec<Vec<f64>> {
        let input = [x];
        eval_function_output_nonzeros(&self.symbolic, &[&input])
    }
}

impl BinaryBundle {
    fn compile(name: &str, builder: BinaryBuilder) -> Self {
        let x = SX::sym(format!("{name}_x"));
        let y = SX::sym(format!("{name}_y"));
        let wrt = SXMatrix::dense_column(vec![x, y]).unwrap();
        let value = SXMatrix::scalar(builder(x, y));
        let gradient = value.gradient(&wrt).unwrap();
        let function = SXFunction::new(
            name,
            vec![NamedMatrix::new("xy", wrt).unwrap()],
            vec![
                NamedMatrix::new("value", value).unwrap(),
                NamedMatrix::new("gradient", gradient).unwrap(),
            ],
        )
        .unwrap();
        let compiled =
            CompiledJitFunction::compile_function(&function, LlvmOptimizationLevel::O2).unwrap();
        Self {
            compiled,
            symbolic: function,
        }
    }

    fn eval(&self, x: f64, y: f64) -> (f64, f64, f64) {
        let mut context = self.compiled.create_context();
        context.input_mut(0).copy_from_slice(&[x, y]);
        self.compiled.eval(&mut context);
        (
            context.output(0)[0],
            context.output(1)[0],
            context.output(1)[1],
        )
    }

    fn eval_symbolic(&self, x: f64, y: f64) -> Vec<Vec<f64>> {
        let input = [x, y];
        eval_function_output_nonzeros(&self.symbolic, &[&input])
    }
}

fn eval_function_output_nonzeros(function: &SXFunction, inputs: &[&[f64]]) -> Vec<Vec<f64>> {
    assert_eq!(function.inputs().len(), inputs.len());
    let mut vars = std::collections::HashMap::new();
    for (input, values) in function.inputs().iter().zip(inputs) {
        assert_eq!(input.matrix().nonzeros().len(), values.len());
        for (&symbol, &value) in input.matrix().nonzeros().iter().zip(values.iter()) {
            vars.insert(symbol.id(), value);
        }
    }
    function
        .outputs()
        .iter()
        .map(|output| {
            output
                .matrix()
                .nonzeros()
                .iter()
                .map(|&expr| eval(expr, &vars))
                .collect()
        })
        .collect()
}

fn sample<T, S>(runner: &mut TestRunner, strategy: S) -> T
where
    S: Strategy<Value = T>,
{
    strategy.new_tree(runner).unwrap().current()
}

fn sample_trig(runner: &mut TestRunner) -> f64 {
    sample(runner, -0.8_f64..0.8)
}

fn sample_positive(runner: &mut TestRunner) -> f64 {
    sample(runner, 0.2_f64..3.0)
}

fn sample_log1p_domain(runner: &mut TestRunner) -> f64 {
    sample(runner, -0.8_f64..3.0)
}

fn sample_acosh_domain(runner: &mut TestRunner) -> f64 {
    sample(runner, 1.1_f64..3.0)
}

fn sample_nonzero(runner: &mut TestRunner) -> f64 {
    sample(runner, prop_oneof![-3.0_f64..-0.2, 0.2..3.0])
}

fn sample_away_from_integer(runner: &mut TestRunner) -> f64 {
    let base = sample(runner, -4_i32..5);
    let frac = sample(runner, prop_oneof![0.15_f64..0.45, 0.55..0.85]);
    f64::from(base) + frac
}

fn sample_away_from_half_integer(runner: &mut TestRunner) -> f64 {
    let base = sample(runner, -4_i32..5);
    let frac = sample(runner, prop_oneof![0.05_f64..0.30, 0.70..0.95]);
    f64::from(base) + frac
}

fn sample_general_pair(runner: &mut TestRunner) -> (f64, f64) {
    (sample(runner, -2.5_f64..2.5), sample(runner, -2.5_f64..2.5))
}

fn sample_division_pair(runner: &mut TestRunner) -> (f64, f64) {
    (sample(runner, -2.5_f64..2.5), sample_nonzero(runner))
}

fn sample_pow_pair(runner: &mut TestRunner) -> (f64, f64) {
    (sample(runner, 0.3_f64..3.0), sample(runner, -1.5_f64..2.0))
}

fn sample_log_base_pair(runner: &mut TestRunner) -> (f64, f64) {
    (
        sample(runner, 0.3_f64..3.0),
        sample(runner, prop_oneof![0.2_f64..0.8, 1.2..3.0]),
    )
}

fn sample_polar_pair(runner: &mut TestRunner) -> (f64, f64) {
    (sample(runner, 0.3_f64..3.0), sample(runner, 0.3_f64..2.5))
}

fn sample_modulo_pair(runner: &mut TestRunner) -> (f64, f64) {
    let bucket = sample(runner, -3_i32..4);
    let frac = sample(runner, 0.2_f64..0.8);
    let y = sample(runner, 0.4_f64..2.0);
    ((f64::from(bucket) + frac) * y, y)
}

fn sample_copysign_pair(runner: &mut TestRunner) -> (f64, f64) {
    (sample(runner, -2.5_f64..2.5), sample_nonzero(runner))
}

fn sample_separated_pair(runner: &mut TestRunner) -> (f64, f64) {
    let x = sample(runner, -2.5_f64..2.5);
    let gap = sample(runner, prop_oneof![-2.0_f64..-0.3, 0.3..2.0]);
    (x, x + gap)
}

fn central_difference_unary(reference: UnaryReference, x: f64, h: f64) -> f64 {
    (reference(x + h) - reference(x - h)) / (2.0 * h)
}

fn central_second_difference_unary(reference: UnaryReference, x: f64, h: f64) -> f64 {
    (reference(x + h) - 2.0 * reference(x) + reference(x - h)) / (h * h)
}

fn central_difference_binary_x(reference: BinaryReference, x: f64, y: f64, h: f64) -> f64 {
    (reference(x + h, y) - reference(x - h, y)) / (2.0 * h)
}

fn central_difference_binary_y(reference: BinaryReference, x: f64, y: f64, h: f64) -> f64 {
    (reference(x, y + h) - reference(x, y - h)) / (2.0 * h)
}

fn assert_close(
    label: &str,
    actual: f64,
    expected: f64,
    abs_tol: f64,
    rel_tol: f64,
    context: &str,
) {
    let delta = (actual - expected).abs();
    let limit = abs_tol.max(rel_tol * expected.abs());
    assert!(
        delta <= limit,
        "{label} {context} mismatch: actual={actual}, expected={expected}, delta={delta}, limit={limit}",
    );
}

fn assert_output_vectors_close(label: &str, actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} {context} output length"
    );
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        assert_close(
            label,
            actual_value,
            expected_value,
            1e-10,
            1e-10,
            &format!("{context} output[{index}]"),
        );
    }
}

fn default_runner() -> TestRunner {
    TestRunner::new(Config::default())
}

fn unary_smooth_cases() -> Vec<UnaryCase> {
    vec![
        UnaryCase {
            label: "sin",
            builder: |x| x.sin(),
            reference: |x| x.sin(),
            sampler: sample_trig,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(5e-3),
            hess_rel_tol: Some(1e-4),
        },
        UnaryCase {
            label: "cos",
            builder: |x| x.cos(),
            reference: |x| x.cos(),
            sampler: sample_trig,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(5e-3),
            hess_rel_tol: Some(1e-4),
        },
        UnaryCase {
            label: "tan",
            builder: |x| x.tan(),
            reference: |x| x.tan(),
            sampler: sample_trig,
            grad_abs_tol: 2e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(2e-2),
            hess_rel_tol: Some(2e-4),
        },
        UnaryCase {
            label: "exp",
            builder: |x| x.exp(),
            reference: |x| x.exp(),
            sampler: sample_trig,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(5e-3),
            hess_rel_tol: Some(1e-4),
        },
        UnaryCase {
            label: "expm1",
            builder: |x| x.expm1(),
            reference: |x| x.exp_m1(),
            sampler: sample_trig,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(5e-3),
            hess_rel_tol: Some(1e-4),
        },
        UnaryCase {
            label: "exp2",
            builder: |x| x.exp2(),
            reference: |x| x.exp2(),
            sampler: sample_trig,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(8e-3),
            hess_rel_tol: Some(1e-4),
        },
        UnaryCase {
            label: "exp10",
            builder: |x| x.exp10(),
            reference: |x| 10.0_f64.powf(x),
            sampler: sample_trig,
            grad_abs_tol: 3e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(3e-2),
            hess_rel_tol: Some(2e-4),
        },
        UnaryCase {
            label: "log",
            builder: |x| x.log(),
            reference: |x| x.ln(),
            sampler: sample_positive,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(1e-2),
            hess_rel_tol: Some(1e-4),
        },
        UnaryCase {
            label: "log1p",
            builder: |x| x.log1p(),
            reference: |x| x.ln_1p(),
            sampler: sample_log1p_domain,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(1e-2),
            hess_rel_tol: Some(1e-4),
        },
        UnaryCase {
            label: "log2",
            builder: |x| x.log2(),
            reference: |x| x.log2(),
            sampler: sample_positive,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(1e-2),
            hess_rel_tol: Some(1e-4),
        },
        UnaryCase {
            label: "log10",
            builder: |x| x.log10(),
            reference: |x| x.log10(),
            sampler: sample_positive,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(1e-2),
            hess_rel_tol: Some(1e-4),
        },
        UnaryCase {
            label: "log_base_3",
            builder: |x| x.log_base(3.0),
            reference: |x| x.ln() / 3.0_f64.ln(),
            sampler: sample_positive,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(1e-2),
            hess_rel_tol: Some(1e-4),
        },
        UnaryCase {
            label: "sqrt",
            builder: |x| x.sqrt(),
            reference: |x| x.sqrt(),
            sampler: sample_positive,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(1e-2),
            hess_rel_tol: Some(1e-4),
        },
        UnaryCase {
            label: "asin",
            builder: |x| x.asin(),
            reference: |x| x.asin(),
            sampler: sample_trig,
            grad_abs_tol: 2e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(2e-2),
            hess_rel_tol: Some(2e-4),
        },
        UnaryCase {
            label: "acos",
            builder: |x| x.acos(),
            reference: |x| x.acos(),
            sampler: sample_trig,
            grad_abs_tol: 2e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(2e-2),
            hess_rel_tol: Some(2e-4),
        },
        UnaryCase {
            label: "atan",
            builder: |x| x.atan(),
            reference: |x| x.atan(),
            sampler: sample_trig,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(1e-2),
            hess_rel_tol: Some(1e-4),
        },
        UnaryCase {
            label: "sinh",
            builder: |x| x.sinh(),
            reference: |x| x.sinh(),
            sampler: sample_trig,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(1e-2),
            hess_rel_tol: Some(1e-4),
        },
        UnaryCase {
            label: "cosh",
            builder: |x| x.cosh(),
            reference: |x| x.cosh(),
            sampler: sample_trig,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(1e-2),
            hess_rel_tol: Some(1e-4),
        },
        UnaryCase {
            label: "tanh",
            builder: |x| x.tanh(),
            reference: |x| x.tanh(),
            sampler: sample_trig,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(1e-2),
            hess_rel_tol: Some(1e-4),
        },
        UnaryCase {
            label: "asinh",
            builder: |x| x.asinh(),
            reference: |x| x.asinh(),
            sampler: sample_trig,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(1e-2),
            hess_rel_tol: Some(1e-4),
        },
        UnaryCase {
            label: "acosh",
            builder: |x| x.acosh(),
            reference: |x| x.acosh(),
            sampler: sample_acosh_domain,
            grad_abs_tol: 2e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(2e-2),
            hess_rel_tol: Some(2e-4),
        },
        UnaryCase {
            label: "atanh",
            builder: |x| x.atanh(),
            reference: |x| x.atanh(),
            sampler: sample_trig,
            grad_abs_tol: 2e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(2e-2),
            hess_rel_tol: Some(2e-4),
        },
        UnaryCase {
            label: "powi_3",
            builder: |x| x.powi(3),
            reference: |x| x.powi(3),
            sampler: sample_general_pair_left_only,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(1e-2),
            hess_rel_tol: Some(1e-4),
        },
        UnaryCase {
            label: "powf_1_5",
            builder: |x| x.powf(1.5),
            reference: |x| x.powf(1.5),
            sampler: sample_positive,
            grad_abs_tol: 2e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: Some(2e-2),
            hess_rel_tol: Some(2e-4),
        },
    ]
}

fn unary_piecewise_cases() -> Vec<UnaryCase> {
    vec![
        UnaryCase {
            label: "abs",
            builder: |x| x.abs(),
            reference: |x| x.abs(),
            sampler: sample_nonzero,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: None,
            hess_rel_tol: None,
        },
        UnaryCase {
            label: "sign",
            builder: |x| x.sign(),
            reference: |x| x.signum(),
            sampler: sample_nonzero,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: None,
            hess_rel_tol: None,
        },
        UnaryCase {
            label: "floor",
            builder: |x| x.floor(),
            reference: |x| x.floor(),
            sampler: sample_away_from_integer,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: None,
            hess_rel_tol: None,
        },
        UnaryCase {
            label: "ceil",
            builder: |x| x.ceil(),
            reference: |x| x.ceil(),
            sampler: sample_away_from_integer,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: None,
            hess_rel_tol: None,
        },
        UnaryCase {
            label: "round",
            builder: |x| x.round(),
            reference: |x| x.round(),
            sampler: sample_away_from_half_integer,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: None,
            hess_rel_tol: None,
        },
        UnaryCase {
            label: "trunc",
            builder: |x| x.trunc(),
            reference: |x| x.trunc(),
            sampler: sample_away_from_integer,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
            hess_abs_tol: None,
            hess_rel_tol: None,
        },
    ]
}

fn unary_smooth_exact_cases() -> Vec<UnaryExactCase> {
    vec![
        UnaryExactCase {
            label: "sin",
            builder: |x| x.sin(),
            point: 0.3,
            value: |x| x.sin(),
            gradient: |x| x.cos(),
            hessian: Some(|x| -x.sin()),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
            hess_abs_tol: Some(1e-10),
            hess_rel_tol: Some(1e-10),
        },
        UnaryExactCase {
            label: "cos",
            builder: |x| x.cos(),
            point: 0.3,
            value: |x| x.cos(),
            gradient: |x| -x.sin(),
            hessian: Some(|x| -x.cos()),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
            hess_abs_tol: Some(1e-10),
            hess_rel_tol: Some(1e-10),
        },
        UnaryExactCase {
            label: "tan",
            builder: |x| x.tan(),
            point: 0.3,
            value: |x| x.tan(),
            gradient: |x| 1.0 / x.cos().powi(2),
            hessian: Some(|x| 2.0 * x.tan() / x.cos().powi(2)),
            grad_abs_tol: 1e-10,
            grad_rel_tol: 1e-10,
            hess_abs_tol: Some(1e-9),
            hess_rel_tol: Some(1e-9),
        },
        UnaryExactCase {
            label: "exp",
            builder: |x| x.exp(),
            point: 0.3,
            value: |x| x.exp(),
            gradient: |x| x.exp(),
            hessian: Some(|x| x.exp()),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
            hess_abs_tol: Some(1e-10),
            hess_rel_tol: Some(1e-10),
        },
        UnaryExactCase {
            label: "expm1",
            builder: |x| x.expm1(),
            point: 0.3,
            value: |x| x.exp_m1(),
            gradient: |x| x.exp(),
            hessian: Some(|x| x.exp()),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
            hess_abs_tol: Some(1e-10),
            hess_rel_tol: Some(1e-10),
        },
        UnaryExactCase {
            label: "exp2",
            builder: |x| x.exp2(),
            point: 0.3,
            value: |x| x.exp2(),
            gradient: |x| x.exp2() * std::f64::consts::LN_2,
            hessian: Some(|x| x.exp2() * std::f64::consts::LN_2.powi(2)),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
            hess_abs_tol: Some(1e-10),
            hess_rel_tol: Some(1e-10),
        },
        UnaryExactCase {
            label: "exp10",
            builder: |x| x.exp10(),
            point: 0.3,
            value: |x| 10.0_f64.powf(x),
            gradient: |x| 10.0_f64.powf(x) * std::f64::consts::LN_10,
            hessian: Some(|x| 10.0_f64.powf(x) * std::f64::consts::LN_10.powi(2)),
            grad_abs_tol: 1e-10,
            grad_rel_tol: 1e-10,
            hess_abs_tol: Some(1e-8),
            hess_rel_tol: Some(1e-9),
        },
        UnaryExactCase {
            label: "log",
            builder: |x| x.log(),
            point: 1.7,
            value: |x| x.ln(),
            gradient: |x| 1.0 / x,
            hessian: Some(|x| -1.0 / x.powi(2)),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
            hess_abs_tol: Some(1e-10),
            hess_rel_tol: Some(1e-10),
        },
        UnaryExactCase {
            label: "log1p",
            builder: |x| x.log1p(),
            point: 0.7,
            value: |x| x.ln_1p(),
            gradient: |x| 1.0 / (1.0 + x),
            hessian: Some(|x| -1.0 / (1.0 + x).powi(2)),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
            hess_abs_tol: Some(1e-10),
            hess_rel_tol: Some(1e-10),
        },
        UnaryExactCase {
            label: "log2",
            builder: |x| x.log2(),
            point: 1.7,
            value: |x| x.log2(),
            gradient: |x| 1.0 / (x * std::f64::consts::LN_2),
            hessian: Some(|x| -1.0 / (x.powi(2) * std::f64::consts::LN_2)),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
            hess_abs_tol: Some(1e-10),
            hess_rel_tol: Some(1e-10),
        },
        UnaryExactCase {
            label: "log10",
            builder: |x| x.log10(),
            point: 1.7,
            value: |x| x.log10(),
            gradient: |x| 1.0 / (x * std::f64::consts::LN_10),
            hessian: Some(|x| -1.0 / (x.powi(2) * std::f64::consts::LN_10)),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
            hess_abs_tol: Some(1e-10),
            hess_rel_tol: Some(1e-10),
        },
        UnaryExactCase {
            label: "sqrt",
            builder: |x| x.sqrt(),
            point: 1.7,
            value: |x| x.sqrt(),
            gradient: |x| 0.5 / x.sqrt(),
            hessian: Some(|x| -0.25 / x.powf(1.5)),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
            hess_abs_tol: Some(1e-10),
            hess_rel_tol: Some(1e-10),
        },
        UnaryExactCase {
            label: "asin",
            builder: |x| x.asin(),
            point: 0.3,
            value: |x| x.asin(),
            gradient: |x| 1.0 / (1.0 - x * x).sqrt(),
            hessian: Some(|x| x / (1.0 - x * x).powf(1.5)),
            grad_abs_tol: 1e-10,
            grad_rel_tol: 1e-10,
            hess_abs_tol: Some(1e-9),
            hess_rel_tol: Some(1e-9),
        },
        UnaryExactCase {
            label: "acos",
            builder: |x| x.acos(),
            point: 0.3,
            value: |x| x.acos(),
            gradient: |x| -1.0 / (1.0 - x * x).sqrt(),
            hessian: Some(|x| -x / (1.0 - x * x).powf(1.5)),
            grad_abs_tol: 1e-10,
            grad_rel_tol: 1e-10,
            hess_abs_tol: Some(1e-9),
            hess_rel_tol: Some(1e-9),
        },
        UnaryExactCase {
            label: "atan",
            builder: |x| x.atan(),
            point: 0.3,
            value: |x| x.atan(),
            gradient: |x| 1.0 / (1.0 + x * x),
            hessian: Some(|x| -2.0 * x / (1.0 + x * x).powi(2)),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
            hess_abs_tol: Some(1e-10),
            hess_rel_tol: Some(1e-10),
        },
        UnaryExactCase {
            label: "sinh",
            builder: |x| x.sinh(),
            point: 0.3,
            value: |x| x.sinh(),
            gradient: |x| x.cosh(),
            hessian: Some(|x| x.sinh()),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
            hess_abs_tol: Some(1e-10),
            hess_rel_tol: Some(1e-10),
        },
        UnaryExactCase {
            label: "cosh",
            builder: |x| x.cosh(),
            point: 0.3,
            value: |x| x.cosh(),
            gradient: |x| x.sinh(),
            hessian: Some(|x| x.cosh()),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
            hess_abs_tol: Some(1e-10),
            hess_rel_tol: Some(1e-10),
        },
        UnaryExactCase {
            label: "tanh",
            builder: |x| x.tanh(),
            point: 0.3,
            value: |x| x.tanh(),
            gradient: |x| 1.0 / x.cosh().powi(2),
            hessian: Some(|x| -2.0 * x.tanh() / x.cosh().powi(2)),
            grad_abs_tol: 1e-10,
            grad_rel_tol: 1e-10,
            hess_abs_tol: Some(1e-9),
            hess_rel_tol: Some(1e-9),
        },
        UnaryExactCase {
            label: "asinh",
            builder: |x| x.asinh(),
            point: 0.3,
            value: |x| x.asinh(),
            gradient: |x| 1.0 / (x * x + 1.0).sqrt(),
            hessian: Some(|x| -x / (x * x + 1.0).powf(1.5)),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
            hess_abs_tol: Some(1e-10),
            hess_rel_tol: Some(1e-10),
        },
        UnaryExactCase {
            label: "acosh",
            builder: |x| x.acosh(),
            point: 1.7,
            value: |x| x.acosh(),
            gradient: |x| 1.0 / ((x - 1.0).sqrt() * (x + 1.0).sqrt()),
            hessian: Some(|x| -x / ((x - 1.0).powf(1.5) * (x + 1.0).powf(1.5))),
            grad_abs_tol: 1e-10,
            grad_rel_tol: 1e-10,
            hess_abs_tol: Some(1e-8),
            hess_rel_tol: Some(1e-8),
        },
        UnaryExactCase {
            label: "atanh",
            builder: |x| x.atanh(),
            point: 0.3,
            value: |x| x.atanh(),
            gradient: |x| 1.0 / (1.0 - x * x),
            hessian: Some(|x| 2.0 * x / (1.0 - x * x).powi(2)),
            grad_abs_tol: 1e-10,
            grad_rel_tol: 1e-10,
            hess_abs_tol: Some(1e-9),
            hess_rel_tol: Some(1e-9),
        },
        UnaryExactCase {
            label: "powi_3",
            builder: |x| x.powi(3),
            point: 1.7,
            value: |x| x.powi(3),
            gradient: |x| 3.0 * x.powi(2),
            hessian: Some(|x| 6.0 * x),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
            hess_abs_tol: Some(1e-10),
            hess_rel_tol: Some(1e-10),
        },
        UnaryExactCase {
            label: "powf_1_5",
            builder: |x| x.powf(1.5),
            point: 1.7,
            value: |x| x.powf(1.5),
            gradient: |x| 1.5 * x.powf(0.5),
            hessian: Some(|x| 0.75 * x.powf(-0.5)),
            grad_abs_tol: 1e-10,
            grad_rel_tol: 1e-10,
            hess_abs_tol: Some(1e-9),
            hess_rel_tol: Some(1e-9),
        },
    ]
}

fn unary_piecewise_exact_cases() -> Vec<UnaryExactCase> {
    vec![
        UnaryExactCase {
            label: "abs",
            builder: |x| x.abs(),
            point: -1.7,
            value: |x| x.abs(),
            gradient: |x| x.signum(),
            hessian: None,
            grad_abs_tol: 1e-12,
            grad_rel_tol: 1e-12,
            hess_abs_tol: None,
            hess_rel_tol: None,
        },
        UnaryExactCase {
            label: "sign",
            builder: |x| x.sign(),
            point: -1.7,
            value: |x| x.signum(),
            gradient: |_| 0.0,
            hessian: None,
            grad_abs_tol: 1e-12,
            grad_rel_tol: 1e-12,
            hess_abs_tol: None,
            hess_rel_tol: None,
        },
        UnaryExactCase {
            label: "floor",
            builder: |x| x.floor(),
            point: 1.3,
            value: |x| x.floor(),
            gradient: |_| 0.0,
            hessian: None,
            grad_abs_tol: 1e-12,
            grad_rel_tol: 1e-12,
            hess_abs_tol: None,
            hess_rel_tol: None,
        },
        UnaryExactCase {
            label: "ceil",
            builder: |x| x.ceil(),
            point: 1.3,
            value: |x| x.ceil(),
            gradient: |_| 0.0,
            hessian: None,
            grad_abs_tol: 1e-12,
            grad_rel_tol: 1e-12,
            hess_abs_tol: None,
            hess_rel_tol: None,
        },
        UnaryExactCase {
            label: "round",
            builder: |x| x.round(),
            point: 1.3,
            value: |x| x.round(),
            gradient: |_| 0.0,
            hessian: None,
            grad_abs_tol: 1e-12,
            grad_rel_tol: 1e-12,
            hess_abs_tol: None,
            hess_rel_tol: None,
        },
        UnaryExactCase {
            label: "trunc",
            builder: |x| x.trunc(),
            point: 1.3,
            value: |x| x.trunc(),
            gradient: |_| 0.0,
            hessian: None,
            grad_abs_tol: 1e-12,
            grad_rel_tol: 1e-12,
            hess_abs_tol: None,
            hess_rel_tol: None,
        },
    ]
}

fn binary_smooth_cases() -> Vec<BinaryCase> {
    vec![
        BinaryCase {
            label: "add",
            builder: |x, y| x + y,
            reference: |x, y| x + y,
            sampler: sample_general_pair,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
        },
        BinaryCase {
            label: "sub",
            builder: |x, y| x - y,
            reference: |x, y| x - y,
            sampler: sample_general_pair,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
        },
        BinaryCase {
            label: "mul",
            builder: |x, y| x * y,
            reference: |x, y| x * y,
            sampler: sample_general_pair,
            grad_abs_tol: 1e-5,
            grad_rel_tol: 1e-5,
        },
        BinaryCase {
            label: "div",
            builder: |x, y| x / y,
            reference: |x, y| x / y,
            sampler: sample_division_pair,
            grad_abs_tol: 2e-5,
            grad_rel_tol: 1e-5,
        },
        BinaryCase {
            label: "pow",
            builder: |x, y| x.pow(y),
            reference: |x, y| x.powf(y),
            sampler: sample_pow_pair,
            grad_abs_tol: 2e-5,
            grad_rel_tol: 1e-5,
        },
        BinaryCase {
            label: "log_base",
            builder: |x, y| x.log_base(y),
            reference: |x, y| x.ln() / y.ln(),
            sampler: sample_log_base_pair,
            grad_abs_tol: 3e-5,
            grad_rel_tol: 1e-5,
        },
        BinaryCase {
            label: "atan2",
            builder: |x, y| x.atan2(y),
            reference: |x, y| x.atan2(y),
            sampler: sample_polar_pair,
            grad_abs_tol: 2e-5,
            grad_rel_tol: 1e-5,
        },
        BinaryCase {
            label: "hypot",
            builder: |x, y| x.hypot(y),
            reference: |x, y| x.hypot(y),
            sampler: sample_polar_pair,
            grad_abs_tol: 2e-5,
            grad_rel_tol: 1e-5,
        },
    ]
}

fn binary_piecewise_cases() -> Vec<BinaryCase> {
    vec![
        BinaryCase {
            label: "mod",
            builder: |x, y| x.modulo(y),
            reference: |x, y| x % y,
            sampler: sample_modulo_pair,
            grad_abs_tol: 3e-5,
            grad_rel_tol: 1e-5,
        },
        BinaryCase {
            label: "copysign",
            builder: |x, y| x.copysign(y),
            reference: |x, y| x.copysign(y),
            sampler: sample_copysign_pair,
            grad_abs_tol: 2e-5,
            grad_rel_tol: 1e-5,
        },
        BinaryCase {
            label: "min",
            builder: |x, y| x.min(y),
            reference: |x, y| x.min(y),
            sampler: sample_separated_pair,
            grad_abs_tol: 2e-5,
            grad_rel_tol: 1e-5,
        },
        BinaryCase {
            label: "max",
            builder: |x, y| x.max(y),
            reference: |x, y| x.max(y),
            sampler: sample_separated_pair,
            grad_abs_tol: 2e-5,
            grad_rel_tol: 1e-5,
        },
    ]
}

fn binary_smooth_exact_cases() -> Vec<BinaryExactCase> {
    vec![
        BinaryExactCase {
            label: "add",
            builder: |x, y| x + y,
            point: (1.7, -0.8),
            value: |x, y| x + y,
            grad_x: |_, _| 1.0,
            grad_y: |_, _| 1.0,
            grad_abs_tol: 1e-12,
            grad_rel_tol: 1e-12,
        },
        BinaryExactCase {
            label: "sub",
            builder: |x, y| x - y,
            point: (1.7, -0.8),
            value: |x, y| x - y,
            grad_x: |_, _| 1.0,
            grad_y: |_, _| -1.0,
            grad_abs_tol: 1e-12,
            grad_rel_tol: 1e-12,
        },
        BinaryExactCase {
            label: "mul",
            builder: |x, y| x * y,
            point: (1.7, -0.8),
            value: |x, y| x * y,
            grad_x: |_, y| y,
            grad_y: |x, _| x,
            grad_abs_tol: 1e-12,
            grad_rel_tol: 1e-12,
        },
        BinaryExactCase {
            label: "div",
            builder: |x, y| x / y,
            point: (1.7, 0.8),
            value: |x, y| x / y,
            grad_x: |_, y| 1.0 / y,
            grad_y: |x, y| -x / y.powi(2),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
        },
        BinaryExactCase {
            label: "pow",
            builder: |x, y| x.pow(y),
            point: (1.7, 0.8),
            value: |x, y| x.powf(y),
            grad_x: |x, y| y * x.powf(y - 1.0),
            grad_y: |x, y| x.powf(y) * x.ln(),
            grad_abs_tol: 1e-10,
            grad_rel_tol: 1e-10,
        },
        BinaryExactCase {
            label: "log_base",
            builder: |x, y| x.log_base(y),
            point: (1.7, 2.3),
            value: |x, y| x.ln() / y.ln(),
            grad_x: |x, y| 1.0 / (x * y.ln()),
            grad_y: |x, y| -x.ln() / (y * y.ln().powi(2)),
            grad_abs_tol: 1e-10,
            grad_rel_tol: 1e-10,
        },
        BinaryExactCase {
            label: "atan2",
            builder: |x, y| x.atan2(y),
            point: (1.7, 0.8),
            value: |x, y| x.atan2(y),
            grad_x: |x, y| y / (x * x + y * y),
            grad_y: |x, y| -x / (x * x + y * y),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
        },
        BinaryExactCase {
            label: "hypot",
            builder: |x, y| x.hypot(y),
            point: (1.7, 0.8),
            value: |x, y| x.hypot(y),
            grad_x: |x, y| x / x.hypot(y),
            grad_y: |x, y| y / x.hypot(y),
            grad_abs_tol: 1e-11,
            grad_rel_tol: 1e-11,
        },
    ]
}

fn binary_piecewise_exact_cases() -> Vec<BinaryExactCase> {
    vec![
        BinaryExactCase {
            label: "mod",
            builder: |x, y| x.modulo(y),
            point: (1.7, 0.8),
            value: |x, y| x % y,
            grad_x: |_, _| 1.0,
            grad_y: |x, y| -(x / y).trunc(),
            grad_abs_tol: 1e-12,
            grad_rel_tol: 1e-12,
        },
        BinaryExactCase {
            label: "copysign",
            builder: |x, y| x.copysign(y),
            point: (-1.7, 0.8),
            value: |x, y| x.copysign(y),
            grad_x: |x, y| x.signum() * y.signum(),
            grad_y: |_, _| 0.0,
            grad_abs_tol: 1e-12,
            grad_rel_tol: 1e-12,
        },
        BinaryExactCase {
            label: "min",
            builder: |x, y| x.min(y),
            point: (1.2, 1.8),
            value: |x, y| x.min(y),
            grad_x: |x, y| if x < y { 1.0 } else { 0.0 },
            grad_y: |x, y| if x < y { 0.0 } else { 1.0 },
            grad_abs_tol: 1e-12,
            grad_rel_tol: 1e-12,
        },
        BinaryExactCase {
            label: "max",
            builder: |x, y| x.max(y),
            point: (1.2, 1.8),
            value: |x, y| x.max(y),
            grad_x: |x, y| if x > y { 1.0 } else { 0.0 },
            grad_y: |x, y| if x > y { 0.0 } else { 1.0 },
            grad_abs_tol: 1e-12,
            grad_rel_tol: 1e-12,
        },
    ]
}

fn sample_general_pair_left_only(runner: &mut TestRunner) -> f64 {
    sample(runner, -2.5_f64..2.5)
}

#[test]
fn jit_unary_smooth_functions_match_numeric_value_gradient_and_hessian() {
    let mut runner = default_runner();
    for case in unary_smooth_cases() {
        let compiled = UnaryBundle::compile(case.label, case.builder, true);
        for _ in 0..SAMPLE_COUNT {
            let point = (case.sampler)(&mut runner);
            let (value, gradient, hessian) = compiled.eval(point);
            let expected_value = (case.reference)(point);
            let expected_gradient = central_difference_unary(case.reference, point, GRAD_STEP);
            let expected_hessian =
                central_second_difference_unary(case.reference, point, HESS_STEP);

            assert_close(case.label, value, expected_value, 1e-10, 1e-10, "value");
            assert_close(
                case.label,
                gradient,
                expected_gradient,
                case.grad_abs_tol,
                case.grad_rel_tol,
                &format!("gradient at x={point}"),
            );
            assert_close(
                case.label,
                hessian.unwrap(),
                expected_hessian,
                case.hess_abs_tol.unwrap(),
                case.hess_rel_tol.unwrap(),
                &format!("hessian at x={point}"),
            );
        }
    }
}

#[test]
fn jit_unary_smooth_functions_match_symbolic_oracle() {
    let mut runner = default_runner();
    for case in unary_smooth_cases() {
        let compiled = UnaryBundle::compile(case.label, case.builder, case.hess_abs_tol.is_some());
        for _ in 0..SAMPLE_COUNT {
            let point = (case.sampler)(&mut runner);
            let symbolic = compiled.eval_symbolic(point);
            let (value, gradient, hessian) = compiled.eval(point);
            assert_output_vectors_close(case.label, &[value], &symbolic[0], "value");
            assert_output_vectors_close(case.label, &[gradient], &symbolic[1], "gradient");
            if let Some(expected_hessian) = symbolic.get(2) {
                assert_output_vectors_close(
                    case.label,
                    &[hessian.unwrap()],
                    expected_hessian,
                    "hessian",
                );
            }
        }
    }
}

#[test]
fn jit_unary_smooth_functions_match_exact_derivatives() {
    for case in unary_smooth_exact_cases() {
        let compiled = UnaryBundle::compile(case.label, case.builder, case.hessian.is_some());
        let (value, gradient, hessian) = compiled.eval(case.point);
        assert_close(
            case.label,
            value,
            (case.value)(case.point),
            1e-12,
            1e-12,
            "value",
        );
        assert_close(
            case.label,
            gradient,
            (case.gradient)(case.point),
            case.grad_abs_tol,
            case.grad_rel_tol,
            &format!("exact gradient at x={}", case.point),
        );
        if let Some(hessian_fn) = case.hessian {
            assert_close(
                case.label,
                hessian.unwrap(),
                hessian_fn(case.point),
                case.hess_abs_tol.unwrap(),
                case.hess_rel_tol.unwrap(),
                &format!("exact hessian at x={}", case.point),
            );
        }
    }
}

#[test]
fn jit_unary_piecewise_functions_match_numeric_value_and_gradient_away_from_kinks() {
    let mut runner = default_runner();
    for case in unary_piecewise_cases() {
        let compiled = UnaryBundle::compile(case.label, case.builder, false);
        for _ in 0..SAMPLE_COUNT {
            let point = (case.sampler)(&mut runner);
            let (value, gradient, _) = compiled.eval(point);
            let expected_value = (case.reference)(point);
            let expected_gradient = central_difference_unary(case.reference, point, GRAD_STEP);

            assert_close(case.label, value, expected_value, 1e-10, 1e-10, "value");
            assert_close(
                case.label,
                gradient,
                expected_gradient,
                case.grad_abs_tol,
                case.grad_rel_tol,
                &format!("gradient at x={point}"),
            );
        }
    }
}

#[test]
fn jit_unary_piecewise_functions_match_exact_derivatives_away_from_kinks() {
    for case in unary_piecewise_exact_cases() {
        let compiled = UnaryBundle::compile(case.label, case.builder, false);
        let (value, gradient, _) = compiled.eval(case.point);
        assert_close(
            case.label,
            value,
            (case.value)(case.point),
            1e-12,
            1e-12,
            "value",
        );
        assert_close(
            case.label,
            gradient,
            (case.gradient)(case.point),
            case.grad_abs_tol,
            case.grad_rel_tol,
            &format!("exact gradient at x={}", case.point),
        );
    }
}

#[test]
fn jit_unary_piecewise_functions_match_symbolic_oracle_away_from_kinks() {
    let mut runner = default_runner();
    for case in unary_piecewise_cases() {
        let compiled = UnaryBundle::compile(case.label, case.builder, false);
        for _ in 0..SAMPLE_COUNT {
            let point = (case.sampler)(&mut runner);
            let symbolic = compiled.eval_symbolic(point);
            let (value, gradient, _) = compiled.eval(point);
            assert_output_vectors_close(case.label, &[value], &symbolic[0], "value");
            assert_output_vectors_close(case.label, &[gradient], &symbolic[1], "gradient");
        }
    }
}

#[test]
fn jit_binary_smooth_functions_match_numeric_gradients() {
    let mut runner = default_runner();
    for case in binary_smooth_cases() {
        let compiled = BinaryBundle::compile(case.label, case.builder);
        for _ in 0..SAMPLE_COUNT {
            let (x, y) = (case.sampler)(&mut runner);
            let (value, grad_x, grad_y) = compiled.eval(x, y);
            let expected_value = (case.reference)(x, y);
            let expected_grad_x = central_difference_binary_x(case.reference, x, y, GRAD_STEP);
            let expected_grad_y = central_difference_binary_y(case.reference, x, y, GRAD_STEP);

            assert_close(case.label, value, expected_value, 1e-10, 1e-10, "value");
            assert_close(
                case.label,
                grad_x,
                expected_grad_x,
                case.grad_abs_tol,
                case.grad_rel_tol,
                &format!("gradient dx at (x={x}, y={y})"),
            );
            assert_close(
                case.label,
                grad_y,
                expected_grad_y,
                case.grad_abs_tol,
                case.grad_rel_tol,
                &format!("gradient dy at (x={x}, y={y})"),
            );
        }
    }
}

#[test]
fn jit_binary_smooth_functions_match_exact_gradients() {
    for case in binary_smooth_exact_cases() {
        let compiled = BinaryBundle::compile(case.label, case.builder);
        let (x, y) = case.point;
        let (value, grad_x, grad_y) = compiled.eval(x, y);
        assert_close(case.label, value, (case.value)(x, y), 1e-12, 1e-12, "value");
        assert_close(
            case.label,
            grad_x,
            (case.grad_x)(x, y),
            case.grad_abs_tol,
            case.grad_rel_tol,
            &format!("exact gradient dx at (x={x}, y={y})"),
        );
        assert_close(
            case.label,
            grad_y,
            (case.grad_y)(x, y),
            case.grad_abs_tol,
            case.grad_rel_tol,
            &format!("exact gradient dy at (x={x}, y={y})"),
        );
    }
}

#[test]
fn jit_binary_smooth_functions_match_symbolic_oracle() {
    let mut runner = default_runner();
    for case in binary_smooth_cases() {
        let compiled = BinaryBundle::compile(case.label, case.builder);
        for _ in 0..SAMPLE_COUNT {
            let (x, y) = (case.sampler)(&mut runner);
            let symbolic = compiled.eval_symbolic(x, y);
            let (value, grad_x, grad_y) = compiled.eval(x, y);
            assert_output_vectors_close(case.label, &[value], &symbolic[0], "value");
            assert_output_vectors_close(case.label, &[grad_x, grad_y], &symbolic[1], "gradient");
        }
    }
}

#[test]
fn jit_binary_piecewise_functions_match_numeric_gradients_away_from_kinks() {
    let mut runner = default_runner();
    for case in binary_piecewise_cases() {
        let compiled = BinaryBundle::compile(case.label, case.builder);
        for _ in 0..SAMPLE_COUNT {
            let (x, y) = (case.sampler)(&mut runner);
            let (value, grad_x, grad_y) = compiled.eval(x, y);
            let expected_value = (case.reference)(x, y);
            let expected_grad_x = central_difference_binary_x(case.reference, x, y, GRAD_STEP);
            let expected_grad_y = central_difference_binary_y(case.reference, x, y, GRAD_STEP);

            assert_close(case.label, value, expected_value, 1e-10, 1e-10, "value");
            assert_close(
                case.label,
                grad_x,
                expected_grad_x,
                case.grad_abs_tol,
                case.grad_rel_tol,
                &format!("gradient dx at (x={x}, y={y})"),
            );
            assert_close(
                case.label,
                grad_y,
                expected_grad_y,
                case.grad_abs_tol,
                case.grad_rel_tol,
                &format!("gradient dy at (x={x}, y={y})"),
            );
        }
    }
}

#[test]
fn jit_binary_piecewise_functions_match_exact_gradients_away_from_kinks() {
    for case in binary_piecewise_exact_cases() {
        let compiled = BinaryBundle::compile(case.label, case.builder);
        let (x, y) = case.point;
        let (value, grad_x, grad_y) = compiled.eval(x, y);
        assert_close(case.label, value, (case.value)(x, y), 1e-12, 1e-12, "value");
        assert_close(
            case.label,
            grad_x,
            (case.grad_x)(x, y),
            case.grad_abs_tol,
            case.grad_rel_tol,
            &format!("exact gradient dx at (x={x}, y={y})"),
        );
        assert_close(
            case.label,
            grad_y,
            (case.grad_y)(x, y),
            case.grad_abs_tol,
            case.grad_rel_tol,
            &format!("exact gradient dy at (x={x}, y={y})"),
        );
    }
}

#[test]
fn jit_binary_piecewise_functions_match_symbolic_oracle_away_from_kinks() {
    let mut runner = default_runner();
    for case in binary_piecewise_cases() {
        let compiled = BinaryBundle::compile(case.label, case.builder);
        for _ in 0..SAMPLE_COUNT {
            let (x, y) = (case.sampler)(&mut runner);
            let symbolic = compiled.eval_symbolic(x, y);
            let (value, grad_x, grad_y) = compiled.eval(x, y);
            assert_output_vectors_close(case.label, &[value], &symbolic[0], "value");
            assert_output_vectors_close(case.label, &[grad_x, grad_y], &symbolic[1], "gradient");
        }
    }
}
