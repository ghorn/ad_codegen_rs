use std::fs;
use std::path::PathBuf;

use anyhow::{Result, bail};
use test_problems::{
    JitOptLevel, ProblemRunOptions, ProblemSpeed, RunRequest, SolverKind, render_markdown_report,
    render_terminal_report, run_cases, write_dashboard, write_html_report, write_json_report,
    write_transcript_artifacts,
};

fn main() -> Result<()> {
    let mut request = RunRequest {
        progress: true,
        ..RunRequest::default()
    };
    let mut output_dir = PathBuf::from("target/test-problems");
    let mut args = std::env::args().skip(1);
    let mut problems = Vec::new();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--problem" => {
                let Some(problem) = args.next() else {
                    bail!("--problem requires an id");
                };
                problems.push(problem);
            }
            "--solver" => {
                let Some(value) = args.next() else {
                    bail!("--solver requires sqp|interior_point|ipopt|both|all");
                };
                request.solvers = match value.as_str() {
                    "sqp" => vec![SolverKind::Sqp],
                    "interior_point" | "ip" => vec![SolverKind::InteriorPoint],
                    #[cfg(feature = "ipopt")]
                    "ipopt" => vec![SolverKind::Ipopt],
                    "both" => vec![SolverKind::Sqp, SolverKind::InteriorPoint],
                    #[cfg(not(feature = "ipopt"))]
                    "all" => vec![SolverKind::Sqp, SolverKind::InteriorPoint],
                    #[cfg(feature = "ipopt")]
                    "all" => vec![
                        SolverKind::Sqp,
                        SolverKind::InteriorPoint,
                        SolverKind::Ipopt,
                    ],
                    _ => bail!("unknown solver: {value}"),
                };
            }
            "--jobs" => {
                let Some(value) = args.next() else {
                    bail!("--jobs requires a value");
                };
                request.jobs = Some(value.parse()?);
            }
            "--problem-set" => {
                let Some(value) = args.next() else {
                    bail!("--problem-set requires fast|slow|all");
                };
                request.problem_set = match value.as_str() {
                    "fast" => Some(ProblemSpeed::Fast),
                    "slow" => Some(ProblemSpeed::Slow),
                    "all" => None,
                    _ => bail!("unknown problem set: {value}"),
                };
            }
            "--jit-opt" => {
                let Some(value) = args.next() else {
                    bail!("--jit-opt requires 0|2|3|s|all");
                };
                let jit_opt_levels = match value.as_str() {
                    "all" => vec![
                        JitOptLevel::O0,
                        JitOptLevel::O2,
                        JitOptLevel::O3,
                        JitOptLevel::Os,
                    ],
                    "0" | "o0" | "O0" => vec![JitOptLevel::O0],
                    "2" | "o2" | "O2" => vec![JitOptLevel::O2],
                    "3" | "o3" | "O3" => vec![JitOptLevel::O3],
                    "s" | "os" | "Os" | "OS" => vec![JitOptLevel::Os],
                    _ => bail!("unknown jit opt level: {value}"),
                };
                request.run_options = jit_opt_levels
                    .into_iter()
                    .map(|jit_opt_level| ProblemRunOptions { jit_opt_level })
                    .collect();
            }
            "--output-dir" => {
                let Some(value) = args.next() else {
                    bail!("--output-dir requires a path");
                };
                output_dir = PathBuf::from(value);
            }
            "--include-skipped" => {
                request.include_skipped = true;
            }
            "--no-progress" => {
                request.progress = false;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => bail!("unknown argument: {arg}"),
        }
    }

    if !problems.is_empty() {
        request.problem_ids = Some(problems);
    }

    fs::create_dir_all(&output_dir)?;
    let mut results = run_cases(&request)?;
    write_transcript_artifacts(&mut results, &output_dir)?;
    let markdown = render_markdown_report(&results);
    let terminal = render_terminal_report(&results);
    let markdown_path = output_dir.join("report.md");
    let html_path = output_dir.join("report.html");
    let json_path = output_dir.join("report.json");
    let dashboard_path = output_dir.join("dashboard.html");
    fs::write(&markdown_path, &markdown)?;
    write_html_report(&results, &html_path)?;
    write_json_report(&results, &json_path)?;
    write_dashboard(&results, &dashboard_path)?;

    println!("{terminal}");
    println!(
        "Wrote:\n- {}\n- {}\n- {}\n- {}",
        markdown_path.display(),
        html_path.display(),
        json_path.display(),
        dashboard_path.display()
    );
    Ok(())
}

fn print_help() {
    println!(
        "Usage: cargo run --release -p test_problems -- [--problem ID]... [--solver sqp|interior_point|ipopt|both|all] [--problem-set fast|slow|all] [--jit-opt 0|2|3|s|all] [--jobs N] [--output-dir DIR] [--include-skipped] [--no-progress]"
    );
}
