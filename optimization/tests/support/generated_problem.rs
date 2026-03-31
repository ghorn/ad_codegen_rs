use anyhow::{Result, anyhow};
use examples_run::{
    casadi_rosenbrock_nlp_constraints_llvm_aot, casadi_rosenbrock_nlp_gradient_llvm_aot,
    casadi_rosenbrock_nlp_jacobian_llvm_aot, casadi_rosenbrock_nlp_lagrangian_hessian_llvm_aot,
    casadi_rosenbrock_nlp_objective_llvm_aot, constrained_rosenbrock_constraints_llvm_aot,
    constrained_rosenbrock_gradient_llvm_aot, constrained_rosenbrock_jacobian_llvm_aot,
    constrained_rosenbrock_lagrangian_hessian_llvm_aot, constrained_rosenbrock_objective_llvm_aot,
    hanging_chain_constraints_llvm_aot, hanging_chain_gradient_llvm_aot,
    hanging_chain_jacobian_llvm_aot, hanging_chain_lagrangian_hessian_llvm_aot,
    hanging_chain_objective_llvm_aot, hs021_gradient_llvm_aot, hs021_inequalities_llvm_aot,
    hs021_inequality_jacobian_llvm_aot, hs021_lagrangian_hessian_llvm_aot,
    hs021_objective_llvm_aot, hs035_gradient_llvm_aot, hs035_inequalities_llvm_aot,
    hs035_inequality_jacobian_llvm_aot, hs035_lagrangian_hessian_llvm_aot,
    hs035_objective_llvm_aot, hs071_equalities_llvm_aot, hs071_equality_jacobian_llvm_aot,
    hs071_gradient_llvm_aot, hs071_inequalities_llvm_aot, hs071_inequality_jacobian_llvm_aot,
    hs071_lagrangian_hessian_llvm_aot, hs071_objective_llvm_aot,
    parameterized_quadratic_equalities_llvm_aot,
    parameterized_quadratic_equality_jacobian_llvm_aot, parameterized_quadratic_gradient_llvm_aot,
    parameterized_quadratic_lagrangian_hessian_llvm_aot,
    parameterized_quadratic_objective_llvm_aot, simple_nlp_constraints_llvm_aot,
    simple_nlp_gradient_llvm_aot, simple_nlp_jacobian_llvm_aot,
    simple_nlp_lagrangian_hessian_llvm_aot, simple_nlp_objective_llvm_aot,
};
use examples_source::{ExampleArtifact, all_examples};
use optimization::{BackendTimingMetadata, CCS, CompiledNlpProblem, ParameterMatrix};
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};
use sx_codegen_llvm::{CompiledJitFunction, JitExecutionContext, LlvmOptimizationLevel};

macro_rules! generated_ccs {
    ($ccs:expr) => {
        CCS::new(
            $ccs.nrow,
            $ccs.ncol,
            $ccs.col_ptrs.to_vec(),
            $ccs.row_indices.to_vec(),
        )
    };
}

pub(crate) type ObjectiveValueFn = dyn for<'a> Fn(&[f64], &[ParameterMatrix<'a>]) -> f64;
pub(crate) type VectorCallbackFn = dyn for<'a> Fn(&[f64], &[ParameterMatrix<'a>], &mut [f64]);
pub(crate) type HessianCallbackFn =
    dyn for<'a> Fn(&[f64], &[ParameterMatrix<'a>], &[f64], &[f64], &mut [f64]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CallbackBackend {
    Aot,
    Jit,
}

impl CallbackBackend {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Aot => "aot",
            Self::Jit => "jit",
        }
    }
}

pub(crate) struct CallbackNlpProblem {
    pub(crate) dimension: usize,
    pub(crate) parameter_ccs: Vec<CCS>,
    pub(crate) equality_jacobian_ccs: CCS,
    pub(crate) inequality_jacobian_ccs: CCS,
    pub(crate) lagrangian_hessian_ccs: CCS,
    pub(crate) backend_timing: BackendTimingMetadata,
    pub(crate) objective_value: Box<ObjectiveValueFn>,
    pub(crate) objective_gradient: Box<VectorCallbackFn>,
    pub(crate) equality_values: Box<VectorCallbackFn>,
    pub(crate) equality_jacobian_values: Box<VectorCallbackFn>,
    pub(crate) inequality_values: Box<VectorCallbackFn>,
    pub(crate) inequality_jacobian_values: Box<VectorCallbackFn>,
    pub(crate) lagrangian_hessian_values: Box<HessianCallbackFn>,
}

impl CompiledNlpProblem for CallbackNlpProblem {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn parameter_count(&self) -> usize {
        self.parameter_ccs.len()
    }

    fn parameter_ccs(&self, parameter_index: usize) -> &CCS {
        &self.parameter_ccs[parameter_index]
    }

    fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        self.backend_timing
    }

    fn equality_count(&self) -> usize {
        self.equality_jacobian_ccs.nrow
    }

    fn inequality_count(&self) -> usize {
        self.inequality_jacobian_ccs.nrow
    }

    fn objective_value(&self, x: &[f64], parameters: &[ParameterMatrix<'_>]) -> f64 {
        (self.objective_value)(x, parameters)
    }

    fn objective_gradient(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        (self.objective_gradient)(x, parameters, out);
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        &self.equality_jacobian_ccs
    }

    fn equality_values(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        (self.equality_values)(x, parameters, out);
    }

    fn equality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        (self.equality_jacobian_values)(x, parameters, out);
    }

    fn inequality_jacobian_ccs(&self) -> &CCS {
        &self.inequality_jacobian_ccs
    }

    fn inequality_values(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        (self.inequality_values)(x, parameters, out);
    }

    fn inequality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        (self.inequality_jacobian_values)(x, parameters, out);
    }

    fn lagrangian_hessian_ccs(&self) -> &CCS {
        &self.lagrangian_hessian_ccs
    }

    fn lagrangian_hessian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        (self.lagrangian_hessian_values)(
            x,
            parameters,
            equality_multipliers,
            inequality_multipliers,
            out,
        );
    }
}

#[derive(Clone, Copy, Debug)]
struct ProblemSpec {
    objective: &'static str,
    gradient: &'static str,
    equalities: Option<&'static str>,
    equality_jacobian: Option<&'static str>,
    inequalities: Option<&'static str>,
    inequality_jacobian: Option<&'static str>,
    lagrangian_hessian: &'static str,
}

#[derive(Clone, Debug)]
struct ExampleFunctionMap {
    creation_time: Duration,
    artifacts: HashMap<String, ExampleArtifact>,
}

struct JitKernel {
    function: CompiledJitFunction,
    context: Mutex<JitExecutionContext>,
}

impl JitKernel {
    fn compile(function: &sx_core::SXFunction) -> Result<Self> {
        let compiled = CompiledJitFunction::compile_function(function, LlvmOptimizationLevel::O3)?;
        let context = Mutex::new(compiled.create_context());
        Ok(Self {
            function: compiled,
            context,
        })
    }

    fn eval_scalar(&self, x: &[f64], parameters: &[ParameterMatrix<'_>]) -> f64 {
        let mut context = lock_context(&self.context);
        load_jit_inputs(&self.function, &mut context, x, parameters, &[], &[]);
        self.function.eval(&mut context);
        context.output(0)[0]
    }

    fn eval_vector(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        let mut context = lock_context(&self.context);
        load_jit_inputs(&self.function, &mut context, x, parameters, &[], &[]);
        self.function.eval(&mut context);
        out.copy_from_slice(context.output(0));
    }

    fn eval_hessian(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        let mut context = lock_context(&self.context);
        load_jit_inputs(
            &self.function,
            &mut context,
            x,
            parameters,
            equality_multipliers,
            inequality_multipliers,
        );
        self.function.eval(&mut context);
        out.copy_from_slice(context.output(0));
    }
}

fn example_function_map() -> Result<&'static ExampleFunctionMap> {
    static CACHE: OnceLock<std::result::Result<ExampleFunctionMap, String>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| {
        let start = Instant::now();
        let artifacts = all_examples()
            .map(|artifacts| {
                artifacts
                    .into_iter()
                    .map(|artifact| (artifact.module_name.clone(), artifact))
                    .collect::<HashMap<_, _>>()
            })
            .map_err(|err| err.to_string())?;
        Ok(ExampleFunctionMap {
            creation_time: start.elapsed(),
            artifacts,
        })
    });
    match cache {
        Ok(map) => Ok(map),
        Err(err) => Err(anyhow!(err.clone())),
    }
}

fn lookup_artifact(module_name: &str) -> Result<ExampleArtifact> {
    example_function_map()?
        .artifacts
        .get(module_name)
        .cloned()
        .ok_or_else(|| anyhow!("missing example artifact {module_name}"))
}

fn lock_context<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poison) => poison.into_inner(),
    }
}

fn load_jit_inputs(
    function: &CompiledJitFunction,
    context: &mut JitExecutionContext,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    equality_multipliers: &[f64],
    inequality_multipliers: &[f64],
) {
    let mut parameter_index = 0;
    for (slot_index, slot) in function.lowered().inputs.iter().enumerate() {
        let input = context.input_mut(slot_index);
        match slot.name.as_str() {
            "x" | "q" => input.copy_from_slice(x),
            "lambda_eq" => input.copy_from_slice(equality_multipliers),
            "mu" => input.copy_from_slice(inequality_multipliers),
            _ => {
                input.copy_from_slice(parameters[parameter_index].values);
                parameter_index += 1;
            }
        }
    }
    debug_assert_eq!(parameter_index, parameters.len());
}

fn ccs_from_output(artifact: &ExampleArtifact) -> CCS {
    let ccs = artifact.function.outputs()[0].matrix().ccs();
    CCS::new(
        ccs.nrow(),
        ccs.ncol(),
        ccs.col_ptrs().to_vec(),
        ccs.row_indices().to_vec(),
    )
}

fn parameter_ccs_from_objective(artifact: &ExampleArtifact) -> Vec<CCS> {
    artifact
        .function
        .inputs()
        .iter()
        .skip(1)
        .map(|input| {
            let ccs = input.matrix().ccs();
            CCS::new(
                ccs.nrow(),
                ccs.ncol(),
                ccs.col_ptrs().to_vec(),
                ccs.row_indices().to_vec(),
            )
        })
        .collect()
}

fn dimension_from_objective(artifact: &ExampleArtifact) -> usize {
    artifact.function.inputs()[0].matrix().nnz()
}

fn build_jit_problem(spec: ProblemSpec) -> Result<CallbackNlpProblem> {
    let creation_time = example_function_map()?.creation_time;
    let objective_artifact = lookup_artifact(spec.objective)?;
    let gradient_artifact = lookup_artifact(spec.gradient)?;
    let equality_artifact = spec.equalities.map(lookup_artifact).transpose()?;
    let equality_jacobian_artifact = spec.equality_jacobian.map(lookup_artifact).transpose()?;
    let inequality_artifact = spec.inequalities.map(lookup_artifact).transpose()?;
    let inequality_jacobian_artifact = spec.inequality_jacobian.map(lookup_artifact).transpose()?;
    let lagrangian_hessian_artifact = lookup_artifact(spec.lagrangian_hessian)?;

    let compile_start = Instant::now();
    let objective_kernel = JitKernel::compile(&objective_artifact.function)?;
    let gradient_kernel = JitKernel::compile(&gradient_artifact.function)?;
    let equality_kernel = equality_artifact
        .as_ref()
        .map(|artifact| JitKernel::compile(&artifact.function))
        .transpose()?;
    let equality_jacobian_kernel = equality_jacobian_artifact
        .as_ref()
        .map(|artifact| JitKernel::compile(&artifact.function))
        .transpose()?;
    let inequality_kernel = inequality_artifact
        .as_ref()
        .map(|artifact| JitKernel::compile(&artifact.function))
        .transpose()?;
    let inequality_jacobian_kernel = inequality_jacobian_artifact
        .as_ref()
        .map(|artifact| JitKernel::compile(&artifact.function))
        .transpose()?;
    let lagrangian_hessian_kernel = JitKernel::compile(&lagrangian_hessian_artifact.function)?;
    let jit_time = compile_start.elapsed();

    let dimension = dimension_from_objective(&objective_artifact);
    let parameter_ccs = parameter_ccs_from_objective(&objective_artifact);
    let equality_jacobian_ccs = equality_jacobian_artifact
        .as_ref()
        .map(ccs_from_output)
        .unwrap_or_else(|| CCS::empty(0, dimension));
    let inequality_jacobian_ccs = inequality_jacobian_artifact
        .as_ref()
        .map(ccs_from_output)
        .unwrap_or_else(|| CCS::empty(0, dimension));
    let lagrangian_hessian_ccs = ccs_from_output(&lagrangian_hessian_artifact);

    Ok(CallbackNlpProblem {
        dimension,
        parameter_ccs,
        equality_jacobian_ccs,
        inequality_jacobian_ccs,
        lagrangian_hessian_ccs,
        backend_timing: BackendTimingMetadata {
            function_creation_time: Some(creation_time),
            derivative_generation_time: None,
            jit_time: Some(jit_time),
        },
        objective_value: Box::new(move |x, parameters| objective_kernel.eval_scalar(x, parameters)),
        objective_gradient: Box::new(move |x, parameters, out| {
            gradient_kernel.eval_vector(x, parameters, out);
        }),
        equality_values: Box::new(move |x, parameters, out| {
            if let Some(kernel) = &equality_kernel {
                kernel.eval_vector(x, parameters, out);
            }
        }),
        equality_jacobian_values: Box::new(move |x, parameters, out| {
            if let Some(kernel) = &equality_jacobian_kernel {
                kernel.eval_vector(x, parameters, out);
            }
        }),
        inequality_values: Box::new(move |x, parameters, out| {
            if let Some(kernel) = &inequality_kernel {
                kernel.eval_vector(x, parameters, out);
            }
        }),
        inequality_jacobian_values: Box::new(move |x, parameters, out| {
            if let Some(kernel) = &inequality_jacobian_kernel {
                kernel.eval_vector(x, parameters, out);
            }
        }),
        lagrangian_hessian_values: Box::new(
            move |x, parameters, equality_multipliers, inequality_multipliers, out| {
                lagrangian_hessian_kernel.eval_hessian(
                    x,
                    parameters,
                    equality_multipliers,
                    inequality_multipliers,
                    out,
                );
            },
        ),
    })
}

fn constrained_rosenbrock_problem_aot() -> CallbackNlpProblem {
    CallbackNlpProblem {
        dimension: 2,
        parameter_ccs: Vec::new(),
        equality_jacobian_ccs: generated_ccs!(
            constrained_rosenbrock_jacobian_llvm_aot::JACOBIAN_CCS
        ),
        inequality_jacobian_ccs: CCS::empty(0, 2),
        lagrangian_hessian_ccs: generated_ccs!(
            constrained_rosenbrock_lagrangian_hessian_llvm_aot::HESSIAN_CCS
        ),
        backend_timing: BackendTimingMetadata::default(),
        objective_value: Box::new(|x, _parameters| {
            let mut objective = 0.0;
            constrained_rosenbrock_objective_llvm_aot::eval(x, &mut objective);
            objective
        }),
        objective_gradient: Box::new(|x, _parameters, out| {
            constrained_rosenbrock_gradient_llvm_aot::eval(x, out);
        }),
        equality_values: Box::new(|x, _parameters, out| {
            constrained_rosenbrock_constraints_llvm_aot::eval(x, &mut out[0]);
        }),
        equality_jacobian_values: Box::new(|x, _parameters, out| {
            constrained_rosenbrock_jacobian_llvm_aot::eval(x, out);
        }),
        inequality_values: Box::new(|_x, _parameters, _out| {}),
        inequality_jacobian_values: Box::new(|_x, _parameters, _out| {}),
        lagrangian_hessian_values: Box::new(
            |x, _parameters, equality_multipliers, _inequality_multipliers, out| {
                constrained_rosenbrock_lagrangian_hessian_llvm_aot::eval(
                    x,
                    equality_multipliers[0],
                    out,
                );
            },
        ),
    }
}

pub(crate) fn constrained_rosenbrock_problem(
    backend: CallbackBackend,
) -> Result<CallbackNlpProblem> {
    match backend {
        CallbackBackend::Aot => Ok(constrained_rosenbrock_problem_aot()),
        CallbackBackend::Jit => build_jit_problem(ProblemSpec {
            objective: "constrained_rosenbrock_objective",
            gradient: "constrained_rosenbrock_gradient",
            equalities: Some("constrained_rosenbrock_constraints"),
            equality_jacobian: Some("constrained_rosenbrock_jacobian"),
            inequalities: None,
            inequality_jacobian: None,
            lagrangian_hessian: "constrained_rosenbrock_lagrangian_hessian",
        }),
    }
}

fn casadi_rosenbrock_nlp_problem_aot() -> CallbackNlpProblem {
    CallbackNlpProblem {
        dimension: 3,
        parameter_ccs: Vec::new(),
        equality_jacobian_ccs: generated_ccs!(
            casadi_rosenbrock_nlp_jacobian_llvm_aot::JACOBIAN_CCS
        ),
        inequality_jacobian_ccs: CCS::empty(0, 3),
        lagrangian_hessian_ccs: generated_ccs!(
            casadi_rosenbrock_nlp_lagrangian_hessian_llvm_aot::HESSIAN_CCS
        ),
        backend_timing: BackendTimingMetadata::default(),
        objective_value: Box::new(|x, _parameters| {
            let mut objective = 0.0;
            casadi_rosenbrock_nlp_objective_llvm_aot::eval(x, &mut objective);
            objective
        }),
        objective_gradient: Box::new(|x, _parameters, out| {
            casadi_rosenbrock_nlp_gradient_llvm_aot::eval(x, out);
        }),
        equality_values: Box::new(|x, _parameters, out| {
            casadi_rosenbrock_nlp_constraints_llvm_aot::eval(x, &mut out[0]);
        }),
        equality_jacobian_values: Box::new(|x, _parameters, out| {
            casadi_rosenbrock_nlp_jacobian_llvm_aot::eval(x, out);
        }),
        inequality_values: Box::new(|_x, _parameters, _out| {}),
        inequality_jacobian_values: Box::new(|_x, _parameters, _out| {}),
        lagrangian_hessian_values: Box::new(
            |x, _parameters, equality_multipliers, _inequality_multipliers, out| {
                casadi_rosenbrock_nlp_lagrangian_hessian_llvm_aot::eval(
                    x,
                    equality_multipliers[0],
                    out,
                );
            },
        ),
    }
}

pub(crate) fn casadi_rosenbrock_nlp_problem(
    backend: CallbackBackend,
) -> Result<CallbackNlpProblem> {
    match backend {
        CallbackBackend::Aot => Ok(casadi_rosenbrock_nlp_problem_aot()),
        CallbackBackend::Jit => build_jit_problem(ProblemSpec {
            objective: "casadi_rosenbrock_nlp_objective",
            gradient: "casadi_rosenbrock_nlp_gradient",
            equalities: Some("casadi_rosenbrock_nlp_constraints"),
            equality_jacobian: Some("casadi_rosenbrock_nlp_jacobian"),
            inequalities: None,
            inequality_jacobian: None,
            lagrangian_hessian: "casadi_rosenbrock_nlp_lagrangian_hessian",
        }),
    }
}

fn simple_nlp_problem_aot() -> CallbackNlpProblem {
    CallbackNlpProblem {
        dimension: 2,
        parameter_ccs: Vec::new(),
        equality_jacobian_ccs: generated_ccs!(simple_nlp_jacobian_llvm_aot::JACOBIAN_CCS),
        inequality_jacobian_ccs: CCS::empty(0, 2),
        lagrangian_hessian_ccs: generated_ccs!(simple_nlp_lagrangian_hessian_llvm_aot::HESSIAN_CCS),
        backend_timing: BackendTimingMetadata::default(),
        objective_value: Box::new(|x, _parameters| {
            let mut objective = 0.0;
            simple_nlp_objective_llvm_aot::eval(x, &mut objective);
            objective
        }),
        objective_gradient: Box::new(|x, _parameters, out| {
            simple_nlp_gradient_llvm_aot::eval(x, out);
        }),
        equality_values: Box::new(|x, _parameters, out| {
            simple_nlp_constraints_llvm_aot::eval(x, &mut out[0]);
        }),
        equality_jacobian_values: Box::new(|x, _parameters, out| {
            simple_nlp_jacobian_llvm_aot::eval(x, out);
        }),
        inequality_values: Box::new(|_x, _parameters, _out| {}),
        inequality_jacobian_values: Box::new(|_x, _parameters, _out| {}),
        lagrangian_hessian_values: Box::new(
            |x, _parameters, equality_multipliers, _inequality_multipliers, out| {
                simple_nlp_lagrangian_hessian_llvm_aot::eval(x, equality_multipliers[0], out);
            },
        ),
    }
}

pub(crate) fn simple_nlp_problem(backend: CallbackBackend) -> Result<CallbackNlpProblem> {
    match backend {
        CallbackBackend::Aot => Ok(simple_nlp_problem_aot()),
        CallbackBackend::Jit => build_jit_problem(ProblemSpec {
            objective: "simple_nlp_objective",
            gradient: "simple_nlp_gradient",
            equalities: Some("simple_nlp_constraints"),
            equality_jacobian: Some("simple_nlp_jacobian"),
            inequalities: None,
            inequality_jacobian: None,
            lagrangian_hessian: "simple_nlp_lagrangian_hessian",
        }),
    }
}

fn hs021_problem_aot() -> CallbackNlpProblem {
    CallbackNlpProblem {
        dimension: 2,
        parameter_ccs: Vec::new(),
        equality_jacobian_ccs: CCS::empty(0, 2),
        inequality_jacobian_ccs: generated_ccs!(hs021_inequality_jacobian_llvm_aot::JACOBIAN_CCS),
        lagrangian_hessian_ccs: generated_ccs!(hs021_lagrangian_hessian_llvm_aot::HESSIAN_CCS),
        backend_timing: BackendTimingMetadata::default(),
        objective_value: Box::new(|x, _parameters| {
            let mut objective = 0.0;
            hs021_objective_llvm_aot::eval(x, &mut objective);
            objective
        }),
        objective_gradient: Box::new(|x, _parameters, out| {
            hs021_gradient_llvm_aot::eval(x, out);
        }),
        equality_values: Box::new(|_x, _parameters, _out| {}),
        equality_jacobian_values: Box::new(|_x, _parameters, _out| {}),
        inequality_values: Box::new(|x, _parameters, out| {
            hs021_inequalities_llvm_aot::eval(x, out);
        }),
        inequality_jacobian_values: Box::new(|x, _parameters, out| {
            hs021_inequality_jacobian_llvm_aot::eval(x, out);
        }),
        lagrangian_hessian_values: Box::new(
            |x, _parameters, _equality_multipliers, inequality_multipliers, out| {
                hs021_lagrangian_hessian_llvm_aot::eval(x, inequality_multipliers, out);
            },
        ),
    }
}

pub(crate) fn hs021_problem(backend: CallbackBackend) -> Result<CallbackNlpProblem> {
    match backend {
        CallbackBackend::Aot => Ok(hs021_problem_aot()),
        CallbackBackend::Jit => build_jit_problem(ProblemSpec {
            objective: "hs021_objective",
            gradient: "hs021_gradient",
            equalities: None,
            equality_jacobian: None,
            inequalities: Some("hs021_inequalities"),
            inequality_jacobian: Some("hs021_inequality_jacobian"),
            lagrangian_hessian: "hs021_lagrangian_hessian",
        }),
    }
}

fn hs035_problem_aot() -> CallbackNlpProblem {
    CallbackNlpProblem {
        dimension: 3,
        parameter_ccs: Vec::new(),
        equality_jacobian_ccs: CCS::empty(0, 3),
        inequality_jacobian_ccs: generated_ccs!(hs035_inequality_jacobian_llvm_aot::JACOBIAN_CCS),
        lagrangian_hessian_ccs: generated_ccs!(hs035_lagrangian_hessian_llvm_aot::HESSIAN_CCS),
        backend_timing: BackendTimingMetadata::default(),
        objective_value: Box::new(|x, _parameters| {
            let mut objective = 0.0;
            hs035_objective_llvm_aot::eval(x, &mut objective);
            objective
        }),
        objective_gradient: Box::new(|x, _parameters, out| {
            hs035_gradient_llvm_aot::eval(x, out);
        }),
        equality_values: Box::new(|_x, _parameters, _out| {}),
        equality_jacobian_values: Box::new(|_x, _parameters, _out| {}),
        inequality_values: Box::new(|x, _parameters, out| {
            hs035_inequalities_llvm_aot::eval(x, out);
        }),
        inequality_jacobian_values: Box::new(|x, _parameters, out| {
            hs035_inequality_jacobian_llvm_aot::eval(x, out);
        }),
        lagrangian_hessian_values: Box::new(
            |x, _parameters, _equality_multipliers, inequality_multipliers, out| {
                hs035_lagrangian_hessian_llvm_aot::eval(x, inequality_multipliers, out);
            },
        ),
    }
}

pub(crate) fn hs035_problem(backend: CallbackBackend) -> Result<CallbackNlpProblem> {
    match backend {
        CallbackBackend::Aot => Ok(hs035_problem_aot()),
        CallbackBackend::Jit => build_jit_problem(ProblemSpec {
            objective: "hs035_objective",
            gradient: "hs035_gradient",
            equalities: None,
            equality_jacobian: None,
            inequalities: Some("hs035_inequalities"),
            inequality_jacobian: Some("hs035_inequality_jacobian"),
            lagrangian_hessian: "hs035_lagrangian_hessian",
        }),
    }
}

fn hs071_problem_aot() -> CallbackNlpProblem {
    CallbackNlpProblem {
        dimension: 4,
        parameter_ccs: Vec::new(),
        equality_jacobian_ccs: generated_ccs!(hs071_equality_jacobian_llvm_aot::JACOBIAN_CCS),
        inequality_jacobian_ccs: generated_ccs!(hs071_inequality_jacobian_llvm_aot::JACOBIAN_CCS),
        lagrangian_hessian_ccs: generated_ccs!(hs071_lagrangian_hessian_llvm_aot::HESSIAN_CCS),
        backend_timing: BackendTimingMetadata::default(),
        objective_value: Box::new(|x, _parameters| {
            let mut objective = 0.0;
            hs071_objective_llvm_aot::eval(x, &mut objective);
            objective
        }),
        objective_gradient: Box::new(|x, _parameters, out| {
            hs071_gradient_llvm_aot::eval(x, out);
        }),
        equality_values: Box::new(|x, _parameters, out| {
            hs071_equalities_llvm_aot::eval(x, &mut out[0]);
        }),
        equality_jacobian_values: Box::new(|x, _parameters, out| {
            hs071_equality_jacobian_llvm_aot::eval(x, out);
        }),
        inequality_values: Box::new(|x, _parameters, out| {
            hs071_inequalities_llvm_aot::eval(x, out);
        }),
        inequality_jacobian_values: Box::new(|x, _parameters, out| {
            hs071_inequality_jacobian_llvm_aot::eval(x, out);
        }),
        lagrangian_hessian_values: Box::new(
            |x, _parameters, equality_multipliers, inequality_multipliers, out| {
                hs071_lagrangian_hessian_llvm_aot::eval(
                    x,
                    equality_multipliers[0],
                    inequality_multipliers,
                    out,
                );
            },
        ),
    }
}

pub(crate) fn hs071_problem(backend: CallbackBackend) -> Result<CallbackNlpProblem> {
    match backend {
        CallbackBackend::Aot => Ok(hs071_problem_aot()),
        CallbackBackend::Jit => build_jit_problem(ProblemSpec {
            objective: "hs071_objective",
            gradient: "hs071_gradient",
            equalities: Some("hs071_equalities"),
            equality_jacobian: Some("hs071_equality_jacobian"),
            inequalities: Some("hs071_inequalities"),
            inequality_jacobian: Some("hs071_inequality_jacobian"),
            lagrangian_hessian: "hs071_lagrangian_hessian",
        }),
    }
}

fn parameterized_quadratic_problem_aot() -> CallbackNlpProblem {
    CallbackNlpProblem {
        dimension: 2,
        parameter_ccs: vec![generated_ccs!(
            parameterized_quadratic_objective_llvm_aot::P_CCS
        )],
        equality_jacobian_ccs: generated_ccs!(
            parameterized_quadratic_equality_jacobian_llvm_aot::JACOBIAN_CCS
        ),
        inequality_jacobian_ccs: CCS::empty(0, 2),
        lagrangian_hessian_ccs: generated_ccs!(
            parameterized_quadratic_lagrangian_hessian_llvm_aot::HESSIAN_CCS
        ),
        backend_timing: BackendTimingMetadata::default(),
        objective_value: Box::new(|x, parameters| {
            let mut objective = 0.0;
            parameterized_quadratic_objective_llvm_aot::eval(
                x,
                parameters[0].values,
                &mut objective,
            );
            objective
        }),
        objective_gradient: Box::new(|x, parameters, out| {
            parameterized_quadratic_gradient_llvm_aot::eval(x, parameters[0].values, out);
        }),
        equality_values: Box::new(|x, parameters, out| {
            parameterized_quadratic_equalities_llvm_aot::eval(x, parameters[0].values, &mut out[0]);
        }),
        equality_jacobian_values: Box::new(|x, parameters, out| {
            parameterized_quadratic_equality_jacobian_llvm_aot::eval(x, parameters[0].values, out);
        }),
        inequality_values: Box::new(|_x, _parameters, _out| {}),
        inequality_jacobian_values: Box::new(|_x, _parameters, _out| {}),
        lagrangian_hessian_values: Box::new(
            |x, parameters, equality_multipliers, _inequality_multipliers, out| {
                parameterized_quadratic_lagrangian_hessian_llvm_aot::eval(
                    x,
                    parameters[0].values,
                    equality_multipliers[0],
                    out,
                );
            },
        ),
    }
}

pub(crate) fn parameterized_quadratic_problem(
    backend: CallbackBackend,
) -> Result<CallbackNlpProblem> {
    match backend {
        CallbackBackend::Aot => Ok(parameterized_quadratic_problem_aot()),
        CallbackBackend::Jit => build_jit_problem(ProblemSpec {
            objective: "parameterized_quadratic_objective",
            gradient: "parameterized_quadratic_gradient",
            equalities: Some("parameterized_quadratic_equalities"),
            equality_jacobian: Some("parameterized_quadratic_equality_jacobian"),
            inequalities: None,
            inequality_jacobian: None,
            lagrangian_hessian: "parameterized_quadratic_lagrangian_hessian",
        }),
    }
}

fn hanging_chain_problem_aot() -> CallbackNlpProblem {
    CallbackNlpProblem {
        dimension: 8,
        parameter_ccs: Vec::new(),
        equality_jacobian_ccs: generated_ccs!(hanging_chain_jacobian_llvm_aot::JACOBIAN_CCS),
        inequality_jacobian_ccs: CCS::empty(0, 8),
        lagrangian_hessian_ccs: generated_ccs!(
            hanging_chain_lagrangian_hessian_llvm_aot::HESSIAN_CCS
        ),
        backend_timing: BackendTimingMetadata::default(),
        objective_value: Box::new(|x, _parameters| {
            let mut objective = 0.0;
            hanging_chain_objective_llvm_aot::eval(x, &mut objective);
            objective
        }),
        objective_gradient: Box::new(|x, _parameters, out| {
            hanging_chain_gradient_llvm_aot::eval(x, out);
        }),
        equality_values: Box::new(|x, _parameters, out| {
            hanging_chain_constraints_llvm_aot::eval(x, out);
        }),
        equality_jacobian_values: Box::new(|x, _parameters, out| {
            hanging_chain_jacobian_llvm_aot::eval(x, out);
        }),
        inequality_values: Box::new(|_x, _parameters, _out| {}),
        inequality_jacobian_values: Box::new(|_x, _parameters, _out| {}),
        lagrangian_hessian_values: Box::new(
            |x, _parameters, equality_multipliers, _inequality_multipliers, out| {
                hanging_chain_lagrangian_hessian_llvm_aot::eval(x, equality_multipliers, out);
            },
        ),
    }
}

pub(crate) fn hanging_chain_problem(backend: CallbackBackend) -> Result<CallbackNlpProblem> {
    match backend {
        CallbackBackend::Aot => Ok(hanging_chain_problem_aot()),
        CallbackBackend::Jit => build_jit_problem(ProblemSpec {
            objective: "hanging_chain_objective",
            gradient: "hanging_chain_gradient",
            equalities: Some("hanging_chain_constraints"),
            equality_jacobian: Some("hanging_chain_jacobian"),
            inequalities: None,
            inequality_jacobian: None,
            lagrangian_hessian: "hanging_chain_lagrangian_hessian",
        }),
    }
}

pub(crate) fn invalid_shape_problem() -> CallbackNlpProblem {
    CallbackNlpProblem {
        dimension: 2,
        parameter_ccs: Vec::new(),
        equality_jacobian_ccs: CCS::empty(0, 2),
        inequality_jacobian_ccs: CCS::empty(1, 1),
        lagrangian_hessian_ccs: generated_ccs!(hs021_lagrangian_hessian_llvm_aot::HESSIAN_CCS),
        backend_timing: BackendTimingMetadata::default(),
        objective_value: Box::new(|_x, _parameters| 0.0),
        objective_gradient: Box::new(|_x, _parameters, out| out.fill(0.0)),
        equality_values: Box::new(|_x, _parameters, _out| {}),
        equality_jacobian_values: Box::new(|_x, _parameters, _out| {}),
        inequality_values: Box::new(|_x, _parameters, out| out[0] = 0.0),
        inequality_jacobian_values: Box::new(|_x, _parameters, out| out.fill(0.0)),
        lagrangian_hessian_values: Box::new(
            |_x, _parameters, _equality_multipliers, _inequality_multipliers, out| {
                out.fill(0.0);
            },
        ),
    }
}

pub(crate) fn parameterized_quadratic_parameter_ccs() -> CCS {
    generated_ccs!(parameterized_quadratic_objective_llvm_aot::P_CCS)
}

pub(crate) fn hanging_chain_initial_guess() -> [f64; 8] {
    [
        0.75,
        0.0,
        1.125,
        -0.6495190528,
        1.875,
        -0.6495190528,
        2.25,
        0.0,
    ]
}
