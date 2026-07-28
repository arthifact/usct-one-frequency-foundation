"""Standard-library command-line interface."""

import argparse
import sys
from pathlib import Path
from textwrap import dedent

from usct.config import load_config, wavelength_diagnostics
from usct.results import save_result
from usct.simulation import simulate_dtn_one_frequency
from usct.verification import VerificationSummary, run_verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m usct",
        description="One-frequency USCT Dirichlet-to-Neumann forward simulator",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("explain", help="explain the modeled experiment")
    commands.add_parser("verify", help="run numerical trust checks")
    simulate = commands.add_parser("simulate", help="run and save one simulation")
    simulate.add_argument("config", type=Path)
    simulate.add_argument("--output", type=Path, required=True)
    plot = commands.add_parser("plot", help="plot an already-saved result")
    plot.add_argument("output", type=Path)
    plot.add_argument("--pattern")
    return parser


def _explain() -> None:
    print(
        dedent(
            """
            USCT one-frequency idealized Dirichlet-to-Neumann experiment

              Driven:    known complex acoustic pressure g(theta) on the full disk boundary
              Equation:  -Laplacian(u) - omega^2/c(x)^2 * (1 + i eta) u = 0
              Condition: u = g on the boundary
              Measured:  outward normal flux du/dn on that same boundary
              Response:  D = I + iQ, with amplitude and phase retained
              Time:      p(x,t) = Re{u(x) exp(-i omega t)}

            With N degrees of freedom, B boundary samples, and P patterns:
              forcing values:       (B, P) complex128
              pressure fields:      (N, P) complex128
              boundary flux result: (B, P) complex128

            This is not a point-source/pressure-receiver model. The hardware source
            and receiver quantities remain to be confirmed before modeling transducers.
            """
        ).strip()
    )


def _print_verification(summary: VerificationSummary) -> None:
    print(f"{'check':30} {'value':>12} {'target':>14} {'status':>8}")
    print("-" * 68)
    for check in summary.checks:
        target = f"{check.relation} {check.threshold:.3e}"
        status = "PASS" if check.passed else "FAIL"
        print(f"{check.name:30} {check.value:12.3e} {target:>14} {status:>8}")
    print("\nAnalytic disk errors by mode (refinement 5):")
    print(f"{'mode':>6} {'field':>12} {'flux':>12} {'orientation':>14}")
    for mode, field, flux, orientation in zip(
        summary.analytic.modes,
        summary.analytic.field_errors,
        summary.analytic.flux_errors,
        summary.analytic.flux_orientation,
        strict=True,
    ):
        print(f"{mode:6d} {field:12.3e} {flux:12.3e} {orientation:14.6f}")
    print("\nThree-mesh convergence (maximum over modes 0, 1, 2):")
    print(f"{'refine':>8} {'dofs':>8} {'field':>12} {'flux':>12}")
    for metrics in summary.convergence:
        print(
            f"{metrics.refinements:8d} {metrics.degrees_of_freedom:8d} "
            f"{max(metrics.field_errors):12.3e} {max(metrics.flux_errors):12.3e}"
        )


def _simulate(config_path: Path, output: Path) -> None:
    config = load_config(config_path)
    diagnostics = wavelength_diagnostics(config)
    print(
        f"mesh refinements={config.domain.refinements}, "
        f"h_max={diagnostics.h_max:.6g}, "
        f"lambda_min={diagnostics.wavelength_minimum:.6g}, "
        f"points/wavelength={diagnostics.points_per_wavelength:.2f} "
        f"(required {diagnostics.required_points_per_wavelength:.2f})"
    )
    result = simulate_dtn_one_frequency(config)
    save_result(result, output)
    report = result.solution.report
    print(
        f"solved {report.degrees_of_freedom} dofs and {report.right_hand_sides} patterns "
        f"with {report.factorizations} LU factorization"
    )
    print(
        f"factorization={report.factorization_time_seconds:.6f}s, "
        f"multi-RHS solve={report.solve_time_seconds:.6f}s, "
        f"max residual={max(report.normalized_interior_residuals):.3e}"
    )
    print(f"wrote {output / 'result.npz'}")
    print(f"wrote {output / 'metadata.json'}")


def main(argv: list[str] | None = None) -> int:
    """Run the selected CLI command and return a process exit status."""

    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "explain":
            _explain()
            return 0
        if arguments.command == "verify":
            summary = run_verification()
            _print_verification(summary)
            return 0 if summary.passed else 1
        if arguments.command == "simulate":
            _simulate(arguments.config, arguments.output)
            return 0
        from usct.plotting import plot_result

        destination = plot_result(arguments.output, arguments.pattern)
        print(f"wrote {destination}")
        return 0
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
