#!/usr/bin/env python3
"""Run the reproducible FieldForge/Crystalite analysis pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from settings import OUTPUT_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent


def _run(script: str, arguments: list[str], env: dict[str, str]) -> None:
    command = [sys.executable, str(SCRIPT_DIR / script), *arguments]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run symmetry, element, Wasserstein, MP20, and MLIP analyses."
    )
    parser.add_argument("--fieldforge-cifs", type=Path, required=True)
    parser.add_argument("--crystalite-cifs", type=Path, required=True)
    parser.add_argument("--mp20-csv", type=Path)
    parser.add_argument("--mlip-json-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--skip",
        action="append",
        choices=["symmetry", "elements", "wasserstein", "mp20", "mlip"],
        default=[],
        help="Skip a pipeline stage. Repeat as needed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    for label, path in [
        ("FieldForge CIF directory", args.fieldforge_cifs),
        ("Crystalite CIF directory", args.crystalite_cifs),
    ]:
        if not path.is_dir():
            raise SystemExit(f"{label} not found: {path}")
    if args.mp20_csv is not None and not args.mp20_csv.is_file():
        raise SystemExit(f"MP20 CSV not found: {args.mp20_csv}")
    if args.mlip_json_dir is not None and not args.mlip_json_dir.is_dir():
        raise SystemExit(f"MLIP JSON directory not found: {args.mlip_json_dir}")

    output = args.output_dir.expanduser().resolve()
    env = os.environ.copy()
    env.update(
        {
            "FIELD_FORGE_CIF_DIR": str(args.fieldforge_cifs.resolve()),
            "CRYSTALITE_CIF_DIR": str(args.crystalite_cifs.resolve()),
            "CRYSTAL_BENCHMARK_OUTPUT_DIR": str(output),
        }
    )
    if args.mp20_csv is not None:
        env["MP20_CSV"] = str(args.mp20_csv.resolve())
    if args.mlip_json_dir is not None:
        env["MLIP_JSON_DIR"] = str(args.mlip_json_dir.resolve())

    common = ["--limit", str(args.limit), "--dpi", str(args.dpi)]
    if "symmetry" not in args.skip:
        _run(
            "plot_crystal_symmetry_distributions.py",
            [
                *common,
                "--workers",
                str(args.workers),
                "--cache-csv",
                str(output / "symmetry/csv/symmetry_records.csv"),
                "--per-model-csv-dir",
                str(output / "symmetry/csv/per_model"),
                "--spacegroup-comparison-output",
                str(output / "symmetry/figures/spacegroup_horizontal_top15.png"),
                "--individual-spacegroup-dir",
                str(output / "symmetry/figures/per_model"),
            ],
            env,
        )
    if "elements" not in args.skip:
        _run(
            "plot_element_distributions.py",
            [*common, "--workers", str(args.workers)],
            env,
        )
    if "wasserstein" not in args.skip:
        if args.mp20_csv is None:
            print("Skipping Wasserstein stage: --mp20-csv was not supplied.")
        else:
            _run("plot_wasserstein_similarity.py", common, env)
    if "mp20" not in args.skip:
        if args.mp20_csv is None:
            print("Skipping MP20 stage: --mp20-csv was not supplied.")
        else:
            _run("add_mp20_reference_outputs.py", ["--dpi", str(args.dpi)], env)
    if "mlip" not in args.skip:
        if args.mlip_json_dir is None:
            print("Skipping MLIP stage: --mlip-json-dir was not supplied.")
        else:
            _run("analyze_mlip_relaxation_metrics.py", common, env)

    print(f"\nPipeline complete. Outputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
