mod dashboard;
mod manifest;
mod model;
mod problems;
mod registry;
mod report;
mod runner;
mod transcript;

pub use dashboard::write_dashboard;
pub use manifest::{
    DEFAULT_MAX_ITERS, IterationLimits, KnownStatus, ProblemManifestEntry, ProblemSpeed,
    manifest_entries, manifest_entry, manifest_entry_by_id, slow_manifest_entry,
};
pub use model::{
    JitOptLevel, ProblemCase, ProblemDescriptor, ProblemRunOptions, ProblemRunRecord, RunArtifacts,
    RunStatus, SolverKind, SolverMetrics, SolverTimingBreakdown, ValidationOutcome,
};
pub use registry::registry;
pub use report::{
    render_html_report, render_markdown_report, render_terminal_report, write_html_report,
    write_json_report,
};
pub use runner::{RunRequest, RunResults, run_cases};
pub use transcript::write_transcript_artifacts;
