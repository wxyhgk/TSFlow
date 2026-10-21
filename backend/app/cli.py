from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from app.core.config import get_settings
from app.contracts.jobs import CalculationParameters, JobCreate
from app.formats.gaussian import (
    DEFAULT_GAUSSIAN_ROUTE,
    gaussian_stem,
    render_gaussian_input,
)
from app.calculators.models import ModelRegistry
from app.formats.sdf import (
    SDFDistanceGeometryError,
    generate_structure_from_2d_sdf,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uma",
        description="Generate a 3D conformer from a 2D SDF, optimize it with UMA/TRIC, and write Gaussian input.",
    )
    parser.add_argument(
        "--service-url",
        default=os.environ.get("UMA_SERVICE_URL", "http://127.0.0.1:8010"),
        help="persistent UMA service URL",
    )
    parser.add_argument(
        "-sdf", "--sdf", action="store_true", help="read one SDF molecule"
    )
    parser.add_argument(
        "-2d",
        "--2d",
        dest="distance_geometry",
        action="store_true",
        help="replace 2D coordinates using deterministic OpenChemLib distance geometry",
    )
    parser.add_argument("input", type=Path, help="input .sdf file")
    parser.add_argument(
        "-o", "--output", type=Path, help="output .gjf path (default: INPUT.gjf)"
    )
    parser.add_argument("--model", default="uma-s-1p2p1", help="local UMA model name")
    parser.add_argument("--charge", type=int, default=0, help="total molecular charge")
    parser.add_argument("--spin", type=int, default=1, help="spin multiplicity")
    parser.add_argument(
        "--fmax", type=float, default=0.05, help="force threshold in eV/A"
    )
    parser.add_argument(
        "--max-steps", type=int, default=100, help="maximum TRIC optimization steps"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="optimization device; the final energy and forces always use CPU",
    )
    parser.add_argument(
        "--route",
        default=DEFAULT_GAUSSIAN_ROUTE,
        help="Gaussian route section",
    )
    parser.add_argument(
        "--nproc", type=int, default=8, help="Gaussian shared CPU count"
    )
    parser.add_argument("--memory", default="8GB", help="Gaussian memory, e.g. 8GB")
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing output file"
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    if not args.sdf or not args.distance_geometry:
        raise ValueError("this workflow requires both -sdf and -2d")
    if args.nproc < 1:
        raise ValueError("--nproc must be at least 1")

    input_path = args.input.expanduser().resolve()
    output_path = (args.output or input_path.with_suffix(".gjf")).expanduser().resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"output already exists: {output_path}; use --force to overwrite"
        )

    print(f"[1/4] Distance geometry: {input_path.name}", file=sys.stderr)
    structure = generate_structure_from_2d_sdf(
        input_path,
        charge=args.charge,
        spin=args.spin,
    )
    print(
        f"      Generated {len(structure.atoms)} atoms (including hydrogens)",
        file=sys.stderr,
    )

    settings = get_settings()
    registry = ModelRegistry(settings.checkpoint_dir)
    model = registry.get(args.model)
    if model is None:
        raise ValueError(f"unknown UMA model: {args.model}")
    if model.engine != "uma":
        raise ValueError(
            f"the local conformer CLI currently supports UMA models only: {args.model}"
        )
    if not model.available:
        raise FileNotFoundError(f"model checkpoint is unavailable: {model.checkpoint}")
    if model.archived:
        raise ValueError(f"model is archived and cannot be used: {args.model}")

    request = JobCreate(
        model_name=args.model,
        task_name="omol",
        calculation="geometry_optimization",
        structure=structure,
        parameters=CalculationParameters(
            fmax=args.fmax, max_steps=args.max_steps, optimization_device=args.device
        ),
    )
    print(
        f"[2/4] UMA/TRIC optimization: model={args.model}, device={args.device}, "
        f"fmax={args.fmax:g} eV/A",
        file=sys.stderr,
    )
    from app.services.cli_service import execute_via_service

    def report(progress) -> None:
        print(
            f"      Step {progress.step:>4}/{progress.max_steps}: "
            f"E={progress.energy_ev:.6f} eV  Fmax={progress.max_force_ev_per_a:.5f} eV/A",
            file=sys.stderr,
        )

    result = execute_via_service(request, args.service_url, report)
    status = "converged" if result.converged else "NOT converged"
    print(
        f"[3/4] CPU verification: {status}; E={result.energy_ev:.10f} eV; "
        f"Fmax={result.max_force_ev_per_a:.6f} eV/A; {result.elapsed_seconds:.2f} s",
        file=sys.stderr,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = render_gaussian_input(
        result.final_structure,
        result,
        stem=gaussian_stem(output_path),
        route=args.route,
        nproc=args.nproc,
        memory=args.memory,
    )
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[4/4] Gaussian input: {output_path}", file=sys.stderr)
    if not result.converged:
        print(
            "WARNING: UMA optimization did not reach the requested force threshold; "
            "the GJF was preserved for inspection.",
            file=sys.stderr,
        )
    return output_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (
        SDFDistanceGeometryError,
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
    ) as exc:
        parser.exit(2, f"uma: error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "uma: interrupted\n")


if __name__ == "__main__":
    main()
