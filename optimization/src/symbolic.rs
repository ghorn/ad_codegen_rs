use std::marker::PhantomData;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use anyhow::Result as AnyResult;
use sx_codegen_llvm::{CompiledJitFunction, JitExecutionContext, LlvmOptimizationLevel};
use sx_core::{CCS as CoreCcs, NamedMatrix, SX, SXFunction, SXMatrix, SxError};
use thiserror::Error;

use crate::{
    BackendTimingMetadata, CCS, ClarabelSqpError, ClarabelSqpOptions, ClarabelSqpSummary,
    CompiledNlpProblem, Index, InteriorPointIterationSnapshot, InteriorPointOptions,
    InteriorPointSolveError, InteriorPointSummary, ParameterMatrix, Vectorize, flatten_value,
    solve_nlp_interior_point, solve_nlp_interior_point_with_callback, solve_nlp_sqp,
    solve_nlp_sqp_with_callback, symbolic_column, symbolic_value,
};
#[cfg(feature = "ipopt")]
use crate::{IpoptOptions, IpoptSolveError, IpoptSummary, solve_nlp_ipopt};

#[derive(Clone, Debug, PartialEq)]
struct SymbolicNlp {
    name: String,
    variables: SXMatrix,
    parameters: Vec<NamedMatrix>,
    objective: SXMatrix,
    equalities: Option<SXMatrix>,
    inequalities: Option<SXMatrix>,
    construction_time: Option<Duration>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymbolicNlpOutputs<E = (), I = ()> {
    pub objective: SX,
    pub equalities: E,
    pub inequalities: I,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypedSymbolicNlp<X, P, E, I> {
    symbolic: SymbolicNlp,
    _marker: TypedMarker<X, P, E, I>,
}

#[derive(Debug)]
struct CompiledJitNlp {
    dimension: Index,
    parameter_ccs: Vec<CCS>,
    equality_jacobian_ccs: CCS,
    inequality_base_jacobian_ccs: CCS,
    lagrangian_hessian_ccs: CCS,
    backend_timing: BackendTimingMetadata,
    objective_value: JitKernel,
    objective_gradient: JitKernel,
    equality_values: Option<JitKernel>,
    equality_jacobian_values: Option<JitKernel>,
    inequality_values: Option<JitKernel>,
    inequality_jacobian_values: Option<JitKernel>,
    lagrangian_hessian_values: JitKernel,
}

#[derive(Debug)]
pub struct TypedCompiledJitNlp<X, P, E, I> {
    inner: CompiledJitNlp,
    _marker: TypedMarker<X, P, E, I>,
}

type TypedMarker<X, P, E, I> = PhantomData<fn() -> (X, P, E, I)>;

pub struct TypedRuntimeNlpBounds<X, I>
where
    X: Vectorize<SX>,
    I: Vectorize<SX>,
{
    pub variable_lower: Option<<X as Vectorize<SX>>::Rebind<f64>>,
    pub variable_upper: Option<<X as Vectorize<SX>>::Rebind<f64>>,
    pub inequality_lower: Option<<I as Vectorize<SX>>::Rebind<f64>>,
    pub inequality_upper: Option<<I as Vectorize<SX>>::Rebind<f64>>,
}

impl<X, I> Default for TypedRuntimeNlpBounds<X, I>
where
    X: Vectorize<SX>,
    I: Vectorize<SX>,
{
    fn default() -> Self {
        Self {
            variable_lower: None,
            variable_upper: None,
            inequality_lower: None,
            inequality_upper: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConstraintBounds {
    pub lower: Option<Vec<f64>>,
    pub upper: Option<Vec<f64>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RuntimeNlpBounds {
    pub variables: ConstraintBounds,
    pub inequalities: ConstraintBounds,
}

#[derive(Debug)]
pub struct RuntimeBoundedJitNlp<'a> {
    base: &'a CompiledJitNlp,
    variable_bounds: ConstraintBounds,
    inequality_mapping: InequalityMapping,
}

#[derive(Debug)]
struct JitKernel {
    function: CompiledJitFunction,
    context: Mutex<JitExecutionContext>,
}

#[derive(Clone, Debug)]
struct InequalityMapping {
    rows: Vec<ConstraintTransform>,
    inequality_jacobian_ccs: CCS,
    inequality_value_map: Vec<JacobianValueMap>,
}

#[derive(Clone, Copy, Debug)]
struct ConstraintTransform {
    source_index: Index,
    sign: f64,
    offset: f64,
}

#[derive(Clone, Copy, Debug)]
struct JacobianValueMap {
    source_value_index: Index,
    sign: f64,
}

#[derive(Debug, Error)]
pub enum SymbolicNlpBuildError {
    #[error("symbolic NLP name cannot be empty")]
    EmptyName,
    #[error(transparent)]
    Graph(#[from] SxError),
}

#[derive(Debug, Error)]
pub enum RuntimeNlpBoundsError {
    #[error(
        "variable bounds length mismatch: expected {expected}, got lower={lower_len}, upper={upper_len}"
    )]
    VariableBoundsLengthMismatch {
        expected: Index,
        lower_len: Index,
        upper_len: Index,
    },
    #[error(
        "constraint bounds length mismatch: expected {expected}, got lower={lower_len}, upper={upper_len}"
    )]
    ConstraintBoundsLengthMismatch {
        expected: Index,
        lower_len: Index,
        upper_len: Index,
    },
    #[error("invalid variable bounds at index {index}: lower={lower} > upper={upper}")]
    InvalidVariableBounds {
        index: Index,
        lower: f64,
        upper: f64,
    },
    #[error("invalid constraint bounds at index {index}: lower={lower} > upper={upper}")]
    InvalidConstraintBounds {
        index: Index,
        lower: f64,
        upper: f64,
    },
}

#[derive(Debug, Error)]
pub enum SymbolicNlpCompileError {
    #[error(transparent)]
    Build(#[from] SymbolicNlpBuildError),
    #[error(transparent)]
    Graph(#[from] SxError),
    #[error("jit compilation failed: {0}")]
    Jit(#[from] anyhow::Error),
}

impl SymbolicNlp {
    pub fn new(
        name: impl Into<String>,
        variables: SXMatrix,
        parameters: Vec<NamedMatrix>,
        objective: SXMatrix,
        equalities: Option<SXMatrix>,
        inequalities: Option<SXMatrix>,
    ) -> Result<Self, SymbolicNlpBuildError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(SymbolicNlpBuildError::EmptyName);
        }

        let objective_function = SXFunction::new(
            format!("{name}_objective_validation"),
            symbolic_inputs(&variables, &parameters)?,
            vec![NamedMatrix::new("objective", objective.clone())?],
        )?;
        debug_assert_eq!(objective_function.n_in(), parameters.len() + 1);
        let _ = objective.scalar_expr()?;

        let equalities = normalize_optional_matrix(equalities);
        if let Some(equalities) = &equalities {
            let validation = SXFunction::new(
                format!("{name}_equalities_validation"),
                symbolic_inputs(&variables, &parameters)?,
                vec![NamedMatrix::new("equalities", equalities.clone())?],
            )?;
            debug_assert_eq!(validation.n_in(), parameters.len() + 1);
        }
        let inequalities = normalize_optional_matrix(inequalities);
        if let Some(inequalities) = &inequalities {
            let validation = SXFunction::new(
                format!("{name}_inequalities_validation"),
                symbolic_inputs(&variables, &parameters)?,
                vec![NamedMatrix::new("inequalities", inequalities.clone())?],
            )?;
            debug_assert_eq!(validation.n_in(), parameters.len() + 1);
        }

        Ok(Self {
            name,
            variables,
            parameters,
            objective,
            equalities,
            inequalities,
            construction_time: None,
        })
    }
}

impl<X, P, E, I> TypedSymbolicNlp<X, P, E, I>
where
    X: Vectorize<SX, Rebind<SX> = X>,
    P: Vectorize<SX, Rebind<SX> = P>,
    E: Vectorize<SX, Rebind<SX> = E>,
    I: Vectorize<SX, Rebind<SX> = I>,
{
    pub fn compile_jit(&self) -> Result<TypedCompiledJitNlp<X, P, E, I>, SymbolicNlpCompileError> {
        self.compile_jit_with_opt_level(LlvmOptimizationLevel::O3)
    }

    pub fn compile_jit_with_opt_level(
        &self,
        opt_level: LlvmOptimizationLevel,
    ) -> Result<TypedCompiledJitNlp<X, P, E, I>, SymbolicNlpCompileError> {
        Ok(TypedCompiledJitNlp {
            inner: compile_symbolic_nlp(&self.symbolic, opt_level)?,
            _marker: PhantomData,
        })
    }
}

pub fn symbolic_nlp<X, P, E, I, F>(
    name: impl Into<String>,
    model: F,
) -> Result<TypedSymbolicNlp<X, P, E, I>, SymbolicNlpBuildError>
where
    X: Vectorize<SX, Rebind<SX> = X>,
    P: Vectorize<SX, Rebind<SX> = P>,
    E: Vectorize<SX, Rebind<SX> = E>,
    I: Vectorize<SX, Rebind<SX> = I>,
    F: FnOnce(&X, &P) -> SymbolicNlpOutputs<E, I>,
{
    let started_at = Instant::now();
    let name = name.into();
    if name.trim().is_empty() {
        return Err(SymbolicNlpBuildError::EmptyName);
    }

    let variables = symbolic_value::<X>("x")?;
    let parameters = symbolic_value::<P>("p")?;
    let outputs = model(&variables, &parameters);

    let variable_matrix = symbolic_column(&variables)?;
    let parameter_matrices = if P::LEN == 0 {
        Vec::new()
    } else {
        vec![NamedMatrix::new("p", symbolic_column(&parameters)?)?]
    };
    let equalities = (E::LEN > 0)
        .then(|| symbolic_column(&outputs.equalities))
        .transpose()?;
    let inequalities = (I::LEN > 0)
        .then(|| symbolic_column(&outputs.inequalities))
        .transpose()?;
    let mut symbolic = SymbolicNlp::new(
        name,
        variable_matrix,
        parameter_matrices,
        SXMatrix::scalar(outputs.objective),
        equalities,
        inequalities,
    )?;
    symbolic.construction_time = Some(started_at.elapsed());
    Ok(TypedSymbolicNlp {
        symbolic,
        _marker: PhantomData,
    })
}

impl CompiledJitNlp {
    fn from_symbolic(
        symbolic: &SymbolicNlp,
        opt_level: LlvmOptimizationLevel,
    ) -> Result<Self, SymbolicNlpCompileError> {
        let derivative_started = Instant::now();
        let functions = derive_symbolic_functions(symbolic)?;
        let derivative_generation_time = derivative_started.elapsed();

        let jit_started = Instant::now();
        let objective_value = JitKernel::compile(&functions.objective_value, opt_level)?;
        let objective_gradient = JitKernel::compile(&functions.objective_gradient, opt_level)?;
        let equality_values = functions
            .equality_values
            .as_ref()
            .map(|function| JitKernel::compile(function, opt_level))
            .transpose()?;
        let equality_jacobian_values = functions
            .equality_jacobian_values
            .as_ref()
            .map(|function| JitKernel::compile(function, opt_level))
            .transpose()?;
        let inequality_values = functions
            .inequality_values
            .as_ref()
            .map(|function| JitKernel::compile(function, opt_level))
            .transpose()?;
        let inequality_jacobian_values = functions
            .inequality_jacobian_values
            .as_ref()
            .map(|function| JitKernel::compile(function, opt_level))
            .transpose()?;
        let lagrangian_hessian_values =
            JitKernel::compile(&functions.lagrangian_hessian_values, opt_level)?;
        let jit_time = jit_started.elapsed();

        Ok(Self {
            dimension: symbolic.variables.nnz(),
            parameter_ccs: symbolic
                .parameters
                .iter()
                .map(|parameter| ccs_from_core(parameter.matrix().ccs()))
                .collect(),
            equality_jacobian_ccs: functions.equality_jacobian_values.as_ref().map_or_else(
                || CCS::empty(0, symbolic.variables.nnz()),
                function_output_ccs,
            ),
            inequality_base_jacobian_ccs: functions.inequality_jacobian_values.as_ref().map_or_else(
                || CCS::empty(0, symbolic.variables.nnz()),
                function_output_ccs,
            ),
            lagrangian_hessian_ccs: function_output_ccs(&functions.lagrangian_hessian_values),
            backend_timing: BackendTimingMetadata {
                function_creation_time: symbolic.construction_time,
                derivative_generation_time: Some(derivative_generation_time),
                jit_time: Some(jit_time),
            },
            objective_value,
            objective_gradient,
            equality_values,
            equality_jacobian_values,
            inequality_values,
            inequality_jacobian_values,
            lagrangian_hessian_values,
        })
    }

    pub fn dimension(&self) -> Index {
        self.dimension
    }

    pub fn parameter_count(&self) -> Index {
        self.parameter_ccs.len()
    }

    pub fn parameter_ccs(&self, parameter_index: Index) -> &CCS {
        &self.parameter_ccs[parameter_index]
    }

    pub fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        self.backend_timing
    }

    pub fn equality_count(&self) -> Index {
        self.equality_jacobian_ccs.nrow
    }

    pub fn inequality_base_count(&self) -> Index {
        self.inequality_base_jacobian_ccs.nrow
    }

    pub fn equality_jacobian_ccs(&self) -> &CCS {
        &self.equality_jacobian_ccs
    }

    pub fn inequality_base_jacobian_ccs(&self) -> &CCS {
        &self.inequality_base_jacobian_ccs
    }

    pub fn lagrangian_hessian_ccs(&self) -> &CCS {
        &self.lagrangian_hessian_ccs
    }

    pub fn objective_value(&self, x: &[f64], parameters: &[ParameterMatrix<'_>]) -> f64 {
        self.objective_value.eval_scalar(x, parameters)
    }

    pub fn objective_gradient(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        self.objective_gradient.eval_vector(x, parameters, out);
    }

    pub fn equality_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        if let Some(kernel) = &self.equality_values {
            kernel.eval_vector(x, parameters, out);
        }
    }

    pub fn equality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        if let Some(kernel) = &self.equality_jacobian_values {
            kernel.eval_vector(x, parameters, out);
        }
    }

    pub fn inequality_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        if let Some(kernel) = &self.inequality_values {
            kernel.eval_vector(x, parameters, out);
        }
    }

    pub fn inequality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        if let Some(kernel) = &self.inequality_jacobian_values {
            kernel.eval_vector(x, parameters, out);
        }
    }

    pub fn lagrangian_hessian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        self.lagrangian_hessian_values
            .eval_hessian(x, parameters, equality_multipliers, inequality_multipliers, out);
    }

    pub fn bind_runtime_bounds(
        &self,
        bounds: RuntimeNlpBounds,
    ) -> Result<RuntimeBoundedJitNlp<'_>, RuntimeNlpBoundsError> {
        let variable_bounds = validate_bound_vectors(self.dimension, bounds.variables, true)?;
        let inequality_bounds =
            validate_bound_vectors(self.inequality_base_count(), bounds.inequalities, false)?;
        let mapping = InequalityMapping::from_runtime_bounds(
            self.inequality_base_jacobian_ccs(),
            &inequality_bounds,
        );
        Ok(RuntimeBoundedJitNlp {
            base: self,
            variable_bounds,
            inequality_mapping: mapping,
        })
    }
}

impl<X, P, E, I> TypedCompiledJitNlp<X, P, E, I>
where
    X: Vectorize<SX, Rebind<SX> = X>,
    P: Vectorize<SX, Rebind<SX> = P>,
    E: Vectorize<SX, Rebind<SX> = E>,
    I: Vectorize<SX, Rebind<SX> = I>,
    <X as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <P as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <I as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
{
    pub fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        self.inner.backend_timing_metadata()
    }

    pub fn bind_runtime_bounds(
        &self,
        bounds: &TypedRuntimeNlpBounds<X, I>,
    ) -> Result<RuntimeBoundedJitNlp<'_>, RuntimeNlpBoundsError> {
        self.inner.bind_runtime_bounds(RuntimeNlpBounds {
            variables: ConstraintBounds {
                lower: bounds.variable_lower.as_ref().map(flatten_value),
                upper: bounds.variable_upper.as_ref().map(flatten_value),
            },
            inequalities: ConstraintBounds {
                lower: bounds.inequality_lower.as_ref().map(flatten_value),
                upper: bounds.inequality_upper.as_ref().map(flatten_value),
            },
        })
    }

    pub fn solve_sqp(
        &self,
        x0: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
        bounds: &TypedRuntimeNlpBounds<X, I>,
        options: &ClarabelSqpOptions,
    ) -> Result<ClarabelSqpSummary, ClarabelSqpError> {
        let bound_problem = self
            .bind_runtime_bounds(bounds)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let x0_values = flatten_value(x0);
        let parameter_values = flatten_value(parameters);
        let parameter_storage = if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: &parameter_values,
            }]
        };
        solve_nlp_sqp(&bound_problem, &x0_values, &parameter_storage, options)
    }

    pub fn solve_sqp_with_callback<CB>(
        &self,
        x0: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
        bounds: &TypedRuntimeNlpBounds<X, I>,
        options: &ClarabelSqpOptions,
        callback: CB,
    ) -> Result<ClarabelSqpSummary, ClarabelSqpError>
    where
        CB: FnMut(&crate::SqpIterationSnapshot),
    {
        let bound_problem = self
            .bind_runtime_bounds(bounds)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let x0_values = flatten_value(x0);
        let parameter_values = flatten_value(parameters);
        let parameter_storage = if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: &parameter_values,
            }]
        };
        solve_nlp_sqp_with_callback(
            &bound_problem,
            &x0_values,
            &parameter_storage,
            options,
            callback,
        )
    }

    pub fn solve_interior_point(
        &self,
        x0: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
        bounds: &TypedRuntimeNlpBounds<X, I>,
        options: &InteriorPointOptions,
    ) -> Result<InteriorPointSummary, InteriorPointSolveError> {
        let bound_problem = self
            .bind_runtime_bounds(bounds)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let x0_values = flatten_value(x0);
        let parameter_values = flatten_value(parameters);
        let parameter_storage = if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: &parameter_values,
            }]
        };
        solve_nlp_interior_point(&bound_problem, &x0_values, &parameter_storage, options)
    }

    pub fn solve_interior_point_with_callback<CB>(
        &self,
        x0: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
        bounds: &TypedRuntimeNlpBounds<X, I>,
        options: &InteriorPointOptions,
        callback: CB,
    ) -> Result<InteriorPointSummary, InteriorPointSolveError>
    where
        CB: FnMut(&InteriorPointIterationSnapshot),
    {
        let bound_problem = self
            .bind_runtime_bounds(bounds)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let x0_values = flatten_value(x0);
        let parameter_values = flatten_value(parameters);
        let parameter_storage = if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: &parameter_values,
            }]
        };
        solve_nlp_interior_point_with_callback(
            &bound_problem,
            &x0_values,
            &parameter_storage,
            options,
            callback,
        )
    }

    #[cfg(feature = "ipopt")]
    pub fn solve_ipopt(
        &self,
        x0: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
        bounds: &TypedRuntimeNlpBounds<X, I>,
        options: &IpoptOptions,
    ) -> Result<IpoptSummary, IpoptSolveError> {
        let bound_problem = self
            .bind_runtime_bounds(bounds)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let x0_values = flatten_value(x0);
        let parameter_values = flatten_value(parameters);
        let parameter_storage = if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: &parameter_values,
            }]
        };
        solve_nlp_ipopt(&bound_problem, &x0_values, &parameter_storage, options)
    }
}

fn compile_symbolic_nlp(
    symbolic: &SymbolicNlp,
    opt_level: LlvmOptimizationLevel,
) -> Result<CompiledJitNlp, SymbolicNlpCompileError> {
    CompiledJitNlp::from_symbolic(symbolic, opt_level)
}

impl CompiledNlpProblem for RuntimeBoundedJitNlp<'_> {
    fn dimension(&self) -> Index {
        self.base.dimension()
    }

    fn parameter_count(&self) -> Index {
        self.base.parameter_count()
    }

    fn parameter_ccs(&self, parameter_index: Index) -> &CCS {
        self.base.parameter_ccs(parameter_index)
    }

    fn variable_bounds(&self, lower: &mut [f64], upper: &mut [f64]) -> bool {
        lower.fill(-crate::NLP_INF);
        upper.fill(crate::NLP_INF);
        if let Some(bounds) = &self.variable_bounds.lower {
            lower.copy_from_slice(bounds);
        }
        if let Some(bounds) = &self.variable_bounds.upper {
            upper.copy_from_slice(bounds);
        }
        true
    }

    fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        self.base.backend_timing_metadata()
    }

    fn equality_count(&self) -> Index {
        self.base.equality_count()
    }

    fn inequality_count(&self) -> Index {
        self.inequality_mapping.rows.len()
    }

    fn objective_value(&self, x: &[f64], parameters: &[ParameterMatrix<'_>]) -> f64 {
        self.base.objective_value(x, parameters)
    }

    fn objective_gradient(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        self.base.objective_gradient(x, parameters, out);
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        self.base.equality_jacobian_ccs()
    }

    fn equality_values(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        self.base.equality_values(x, parameters, out);
    }

    fn equality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        self.base.equality_jacobian_values(x, parameters, out);
    }

    fn inequality_jacobian_ccs(&self) -> &CCS {
        &self.inequality_mapping.inequality_jacobian_ccs
    }

    fn inequality_values(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        let mut inequality_values = vec![0.0; self.base.inequality_base_count()];
        self.base
            .inequality_values(x, parameters, &mut inequality_values);
        for (slot, transform) in out.iter_mut().zip(self.inequality_mapping.rows.iter()) {
            *slot = transform.sign * inequality_values[transform.source_index] + transform.offset;
        }
    }

    fn inequality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        let mut source_values = vec![0.0; self.base.inequality_base_jacobian_ccs().nnz()];
        self.base
            .inequality_jacobian_values(x, parameters, &mut source_values);
        for (slot, mapping) in out.iter_mut().zip(self.inequality_mapping.inequality_value_map.iter()) {
            *slot = mapping.sign * source_values[mapping.source_value_index];
        }
    }

    fn lagrangian_hessian_ccs(&self) -> &CCS {
        self.base.lagrangian_hessian_ccs()
    }

    fn lagrangian_hessian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        let mut base_inequality_multipliers = vec![0.0; self.base.inequality_base_count()];
        for (multiplier, transform) in inequality_multipliers
            .iter()
            .zip(self.inequality_mapping.rows.iter())
        {
            base_inequality_multipliers[transform.source_index] += transform.sign * multiplier;
        }
        self.base
            .lagrangian_hessian_values(x, parameters, equality_multipliers, &base_inequality_multipliers, out);
    }
}

impl InequalityMapping {
    fn from_runtime_bounds(base_ccs: &CCS, bounds: &ConstraintBounds) -> Self {
        let lower = bounds.lower.as_deref().unwrap_or(&[]).to_vec();
        let upper = bounds.upper.as_deref().unwrap_or(&[]).to_vec();
        let count = base_ccs.nrow;
        let mut inequality_rows = Vec::new();
        let mut inequality_by_source = vec![Vec::<(Index, f64)>::new(); count];

        for (source_index, rows_for_source) in inequality_by_source
            .iter_mut()
            .enumerate()
            .take(count)
        {
            let lower_bound = lower.get(source_index).copied().unwrap_or(-crate::NLP_INF);
            let upper_bound = upper.get(source_index).copied().unwrap_or(crate::NLP_INF);
            if lower_bound == -crate::NLP_INF && upper_bound == crate::NLP_INF {
                continue;
            }
            if lower_bound > -crate::NLP_INF {
                let row = inequality_rows.len();
                inequality_rows.push(ConstraintTransform {
                    source_index,
                    sign: -1.0,
                    offset: lower_bound,
                });
                rows_for_source.push((row, -1.0));
            }
            if upper_bound < crate::NLP_INF {
                let row = inequality_rows.len();
                inequality_rows.push(ConstraintTransform {
                    source_index,
                    sign: 1.0,
                    offset: -upper_bound,
                });
                rows_for_source.push((row, 1.0));
            }
        }

        let (inequality_jacobian_ccs, inequality_value_map) =
            remap_constraint_jacobian(base_ccs, &inequality_by_source);
        Self {
            rows: inequality_rows,
            inequality_jacobian_ccs,
            inequality_value_map,
        }
    }
}

impl JitKernel {
    fn compile(function: &SXFunction, opt_level: LlvmOptimizationLevel) -> AnyResult<Self> {
        let compiled = CompiledJitFunction::compile_function(function, opt_level)?;
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

fn symbolic_inputs(
    variables: &SXMatrix,
    parameters: &[NamedMatrix],
) -> Result<Vec<NamedMatrix>, SymbolicNlpBuildError> {
    let mut inputs = Vec::with_capacity(parameters.len() + 1);
    inputs.push(NamedMatrix::new("x", variables.clone())?);
    inputs.extend(parameters.iter().cloned());
    Ok(inputs)
}

fn normalize_optional_matrix(matrix: Option<SXMatrix>) -> Option<SXMatrix> {
    match matrix {
        Some(matrix) if matrix.nnz() == 0 => None,
        other => other,
    }
}

fn function_output_ccs(function: &SXFunction) -> CCS {
    ccs_from_core(function.outputs()[0].matrix().ccs())
}

struct DerivedSymbolicFunctions {
    objective_value: SXFunction,
    objective_gradient: SXFunction,
    equality_values: Option<SXFunction>,
    equality_jacobian_values: Option<SXFunction>,
    inequality_values: Option<SXFunction>,
    inequality_jacobian_values: Option<SXFunction>,
    lagrangian_hessian_values: SXFunction,
}

fn derive_symbolic_functions(
    symbolic: &SymbolicNlp,
) -> Result<DerivedSymbolicFunctions, SymbolicNlpCompileError> {
    let base_inputs = symbolic_inputs(&symbolic.variables, &symbolic.parameters)?;
    let objective_value = SXFunction::new(
        format!("{}_objective", symbolic.name),
        base_inputs.clone(),
        vec![NamedMatrix::new("objective", symbolic.objective.clone())?],
    )?;
    let gradient = symbolic.objective.gradient(&symbolic.variables)?;
    let objective_gradient = SXFunction::new(
        format!("{}_gradient", symbolic.name),
        base_inputs.clone(),
        vec![NamedMatrix::new("gradient", gradient)?],
    )?;

    let equality_values = symbolic
        .equalities
        .as_ref()
        .map(|equalities| {
            SXFunction::new(
                format!("{}_equalities", symbolic.name),
                base_inputs.clone(),
                vec![NamedMatrix::new("equalities", equalities.clone())?],
            )
        })
        .transpose()?;
    let equality_jacobian_values = symbolic
        .equalities
        .as_ref()
        .map(|equalities| {
            SXFunction::new(
                format!("{}_equality_jacobian", symbolic.name),
                base_inputs.clone(),
                vec![NamedMatrix::new(
                    "equality_jacobian",
                    equalities.jacobian(&symbolic.variables)?,
                )?],
            )
        })
        .transpose()?;
    let inequality_values = symbolic
        .inequalities
        .as_ref()
        .map(|inequalities| {
            SXFunction::new(
                format!("{}_inequalities", symbolic.name),
                base_inputs.clone(),
                vec![NamedMatrix::new("inequalities", inequalities.clone())?],
            )
        })
        .transpose()?;
    let inequality_jacobian_values = symbolic
        .inequalities
        .as_ref()
        .map(|inequalities| {
            SXFunction::new(
                format!("{}_inequality_jacobian", symbolic.name),
                base_inputs.clone(),
                vec![NamedMatrix::new(
                    "inequality_jacobian",
                    inequalities.jacobian(&symbolic.variables)?,
                )?],
            )
        })
        .transpose()?;

    let mut hessian_inputs = base_inputs.clone();
    let mut lagrangian = symbolic.objective.scalar_expr()?;
    let equality_count = symbolic.equalities.as_ref().map_or(0, SXMatrix::nnz);
    if let Some(equalities) = &symbolic.equalities {
        let lambda = SXMatrix::sym("lambda_equalities", CoreCcs::column_vector(equality_count)?)?;
        for idx in 0..equalities.nnz() {
            lagrangian += lambda.nz(idx) * equalities.nz(idx);
        }
        hessian_inputs.push(NamedMatrix::new("lambda_equalities", lambda)?);
    }
    let inequality_count = symbolic.inequalities.as_ref().map_or(0, SXMatrix::nnz);
    if let Some(inequalities) = &symbolic.inequalities {
        let lambda =
            SXMatrix::sym("lambda_inequalities", CoreCcs::column_vector(inequality_count)?)?;
        for idx in 0..inequalities.nnz() {
            lagrangian += lambda.nz(idx) * inequalities.nz(idx);
        }
        hessian_inputs.push(NamedMatrix::new("lambda_inequalities", lambda)?);
    }
    let lagrangian_hessian = SXMatrix::scalar(lagrangian).hessian(&symbolic.variables)?;
    let lagrangian_hessian_values = SXFunction::new(
        format!("{}_lagrangian_hessian", symbolic.name),
        hessian_inputs,
        vec![NamedMatrix::new("lagrangian_hessian", lagrangian_hessian)?],
    )?;

    Ok(DerivedSymbolicFunctions {
        objective_value,
        objective_gradient,
        equality_values,
        equality_jacobian_values,
        inequality_values,
        inequality_jacobian_values,
        lagrangian_hessian_values,
    })
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
            "x" => input.copy_from_slice(x),
            "lambda_equalities" => input.copy_from_slice(equality_multipliers),
            "lambda_inequalities" => input.copy_from_slice(inequality_multipliers),
            _ => {
                input.copy_from_slice(parameters[parameter_index].values);
                parameter_index += 1;
            }
        }
    }
    debug_assert_eq!(parameter_index, parameters.len());
}

fn ccs_from_core(ccs: &CoreCcs) -> CCS {
    CCS::new(
        ccs.nrow(),
        ccs.ncol(),
        ccs.col_ptrs().to_vec(),
        ccs.row_indices().to_vec(),
    )
}

fn validate_bound_vectors(
    expected: Index,
    bounds: ConstraintBounds,
    is_variable: bool,
) -> Result<ConstraintBounds, RuntimeNlpBoundsError> {
    let lower_len = bounds.lower.as_ref().map_or(expected, Vec::len);
    let upper_len = bounds.upper.as_ref().map_or(expected, Vec::len);
    if bounds
        .lower
        .as_ref()
        .is_some_and(|values| values.len() != expected)
        || bounds
            .upper
            .as_ref()
            .is_some_and(|values| values.len() != expected)
    {
        return Err(if is_variable {
            RuntimeNlpBoundsError::VariableBoundsLengthMismatch {
                expected,
                lower_len,
                upper_len,
            }
        } else {
            RuntimeNlpBoundsError::ConstraintBoundsLengthMismatch {
                expected,
                lower_len,
                upper_len,
            }
        });
    }

    if let (Some(lower), Some(upper)) = (&bounds.lower, &bounds.upper) {
        for (index, (&lower, &upper)) in lower.iter().zip(upper.iter()).enumerate() {
            if lower > upper {
                return Err(if is_variable {
                    RuntimeNlpBoundsError::InvalidVariableBounds {
                        index,
                        lower,
                        upper,
                    }
                } else {
                    RuntimeNlpBoundsError::InvalidConstraintBounds {
                        index,
                        lower,
                        upper,
                    }
                });
            }
        }
    }
    Ok(bounds)
}

fn remap_constraint_jacobian(
    base_ccs: &CCS,
    rows_by_source: &[Vec<(Index, f64)>],
) -> (CCS, Vec<JacobianValueMap>) {
    let mut col_ptrs = Vec::with_capacity(base_ccs.ncol + 1);
    let mut row_indices = Vec::new();
    let mut value_map = Vec::new();
    col_ptrs.push(0);
    for col in 0..base_ccs.ncol {
        for source_value_index in base_ccs.col_ptrs[col]..base_ccs.col_ptrs[col + 1] {
            let source_row = base_ccs.row_indices[source_value_index];
            for &(output_row, sign) in &rows_by_source[source_row] {
                row_indices.push(output_row);
                value_map.push(JacobianValueMap {
                    source_value_index,
                    sign,
                });
            }
        }
        col_ptrs.push(row_indices.len());
    }
    (
        CCS::new(
            rows_by_source.iter().flatten().count(),
            base_ccs.ncol,
            col_ptrs,
            row_indices,
        ),
        value_map,
    )
}
