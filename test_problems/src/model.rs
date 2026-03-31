use std::time::Duration;

use optimization::LlvmOptimizationLevel;
use serde::{Serialize, ser::SerializeStruct};

use crate::manifest::KnownStatus;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverKind {
    Sqp,
    InteriorPoint,
    #[cfg(feature = "ipopt")]
    Ipopt,
}

impl SolverKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Sqp => "sqp",
            Self::InteriorPoint => "interior_point",
            #[cfg(feature = "ipopt")]
            Self::Ipopt => "ipopt",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JitOptLevel {
    O0,
    O2,
    #[default]
    O3,
    Os,
}

impl JitOptLevel {
    pub const fn label(self) -> &'static str {
        match self {
            Self::O0 => "O0",
            Self::O2 => "O2",
            Self::O3 => "O3",
            Self::Os => "Os",
        }
    }

    pub const fn into_llvm(self) -> LlvmOptimizationLevel {
        match self {
            Self::O0 => LlvmOptimizationLevel::O0,
            Self::O2 => LlvmOptimizationLevel::O2,
            Self::O3 => LlvmOptimizationLevel::O3,
            Self::Os => LlvmOptimizationLevel::Os,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
pub struct ProblemRunOptions {
    pub jit_opt_level: JitOptLevel,
}

impl ProblemRunOptions {
    pub const fn label(self) -> &'static str {
        self.jit_opt_level.label()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Passed,
    ReducedAccuracy,
    FailedValidation,
    SolveError,
    Skipped,
}

impl RunStatus {
    pub const fn accepted(self) -> bool {
        matches!(self, Self::Passed | Self::ReducedAccuracy)
    }

    pub const fn failed(self) -> bool {
        matches!(self, Self::FailedValidation | Self::SolveError)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationTier {
    #[default]
    Failed,
    Passed,
    ReducedAccuracy,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProblemDescriptor {
    pub id: String,
    pub family: String,
    pub variant: String,
    pub source: String,
    pub description: String,
    pub parameterized: bool,
    pub num_vars: usize,
    pub num_eq: usize,
    pub num_ineq: usize,
    pub num_box: usize,
    pub dof: usize,
    pub constrained: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SolverMetrics {
    pub iterations: Option<usize>,
    pub objective: Option<f64>,
    pub equality_inf: Option<f64>,
    pub inequality_inf: Option<f64>,
    pub primal_inf: Option<f64>,
    pub dual_inf: Option<f64>,
    pub complementarity_inf: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SolverTimingBreakdown {
    pub function_creation_time: Option<Duration>,
    pub derivative_generation_time: Option<Duration>,
    pub jit_time: Option<Duration>,
    pub compile_wall_time: Option<Duration>,
    pub solve_time: Option<Duration>,
    pub total_wall_time: Duration,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ValidationOutcome {
    pub tier: ValidationTier,
    pub tolerance: String,
    pub detail: String,
}

impl ValidationOutcome {
    pub const fn passed(&self) -> bool {
        matches!(
            self.tier,
            ValidationTier::Passed | ValidationTier::ReducedAccuracy
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProblemRunRecord {
    pub id: String,
    pub solver: SolverKind,
    pub options: ProblemRunOptions,
    pub expected: KnownStatus,
    pub max_iters_limit: usize,
    pub status: RunStatus,
    pub descriptor: ProblemDescriptor,
    pub solution: Option<Vec<f64>>,
    pub metrics: SolverMetrics,
    pub timing: SolverTimingBreakdown,
    pub validation: ValidationOutcome,
    pub solver_thresholds: Option<String>,
    pub error: Option<String>,
    #[serde(skip)]
    pub console_output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub console_output_path: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunArtifacts {
    pub markdown_report: String,
    pub json_report: String,
    pub dashboard_html: String,
}

pub struct ProblemCase {
    pub id: &'static str,
    pub family: &'static str,
    pub variant: &'static str,
    pub source: &'static str,
    pub description: &'static str,
    pub parameterized: bool,
    pub(crate) run_fn: Box<
        dyn Fn(SolverKind, ProblemRunOptions, usize, KnownStatus) -> ProblemRunRecord + Send + Sync,
    >,
}

impl ProblemCase {
    pub fn run(
        &self,
        solver: SolverKind,
        options: ProblemRunOptions,
        max_iters_limit: usize,
        expected: KnownStatus,
    ) -> ProblemRunRecord {
        (self.run_fn)(solver, options, max_iters_limit, expected)
    }
}

impl std::fmt::Debug for ProblemCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProblemCase")
            .field("id", &self.id)
            .field("family", &self.family)
            .field("variant", &self.variant)
            .field("source", &self.source)
            .field("description", &self.description)
            .field("parameterized", &self.parameterized)
            .finish_non_exhaustive()
    }
}

impl Serialize for SolverTimingBreakdown {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("SolverTimingBreakdown", 6)?;
        state.serialize_field(
            "function_creation_time",
            &self
                .function_creation_time
                .map(|duration| duration.as_secs_f64()),
        )?;
        state.serialize_field(
            "derivative_generation_time",
            &self
                .derivative_generation_time
                .map(|duration| duration.as_secs_f64()),
        )?;
        state.serialize_field(
            "jit_time",
            &self.jit_time.map(|duration| duration.as_secs_f64()),
        )?;
        state.serialize_field(
            "compile_wall_time",
            &self
                .compile_wall_time
                .map(|duration| duration.as_secs_f64()),
        )?;
        state.serialize_field(
            "solve_time",
            &self.solve_time.map(|duration| duration.as_secs_f64()),
        )?;
        state.serialize_field("total_wall_time", &self.total_wall_time.as_secs_f64())?;
        state.end()
    }
}
