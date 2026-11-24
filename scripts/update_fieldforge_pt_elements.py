#!/usr/bin/env python3
"""Replace only FieldForge (PT) CIF-derived element statistics.

The existing cache supplies the other models.  This keeps their composition
rows unchanged while rebuilding the merged element figures and the MP20
element comparison from the updated FieldForge cache.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import plot_element_distributions as elements
import add_mp20_reference_outputs as mp20_outputs
from settings import FIELD_FORGE_CIF_DIR, FIELD_FORGE_NAME, OUTPUT_ROOT


MODEL = FIELD_FORGE_NAME
DEFAULT_CIF_DIR = FIELD_FORGE_CIF_DIR
DEFAULT_CACHE = OUTPUT_ROOT / "elements/csv/structure_element_cache.csv"
DEFAULT_OUTPUT = OUTPUT_ROOT / "elements"
DEFAULT_MP20_OUTPUT = OUTPUT_ROOT / "mp20"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update only FieldForge (PT) in CIF-derived element analysis."
    )
    parser.add_argument("--cif-dir", type=Path, default=DEFAULT_CIF_DIR)
    parser.add_argument("--cache-csv", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mp20-output-dir", type=Path, default=DEFAULT_MP20_OUTPUT)
    parser.add_argument(
        "--mp20-element-cache-csv",
        type=Path,
        default=OUTPUT_ROOT / "mp20/csv/mp20_structure_element_cache.csv",
    )
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--top-elements", type=int, default=20)
    parser.add_argument(
        "--periodic-only",
        action="store_true",
        help="Update the composition cache and only the FieldForge periodic-table plots.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.cif_dir.is_dir():
        raise SystemExit(f"FieldForge CIF directory not found: {args.cif_dir}")
    if not args.cache_csv.exists():
        raise SystemExit(f"Existing element cache not found: {args.cache_csv}")

    old_rows = elements.read_structure_cache(args.cache_csv)
    model_order = []
    for row in old_rows:
        if row["model"] not in model_order:
            model_order.append(row["model"])
    if MODEL not in model_order:
        model_order.insert(0, MODEL)

    new_rows = elements.analyze_models(
        [elements.ModelInput(MODEL, args.cif_dir)],
        recursive=True,
        limit=args.limit,
        workers=args.workers,
    )
    merged_rows = [row for row in old_rows if row["model"] != MODEL] + new_rows
    # Restore the original model order in the cache and all downstream plots.
    merged_rows.sort(key=lambda row: (model_order.index(row["model"]), row["file"]))
    elements.write_structure_cache(merged_rows, args.cache_csv)

    stats = elements.build_element_stats(merged_rows, model_order)
    if args.periodic_only:
        fieldforge_stats = stats[stats["model"] == MODEL]
        elements.plot_periodic_tables(fieldforge_stats, [MODEL], args.output_dir, args.dpi)
    else:
        elements.write_element_stats(stats, model_order, args.output_dir)
        elements.plot_top_element_comparison(
            stats, model_order, args.output_dir, args.top_elements, args.dpi
        )
        elements.plot_unique_elements_distribution(merged_rows, model_order, args.output_dir, args.dpi)
        elements.plot_periodic_tables(stats, model_order, args.output_dir, args.dpi)

    if args.mp20_element_cache_csv.exists():
        if args.periodic_only:
            mp20_rows = elements.read_structure_cache(args.mp20_element_cache_csv)
            mp20_fieldforge_stats = elements.build_element_stats(
                mp20_rows + new_rows, ["MP20", MODEL]
            )
            elements.plot_periodic_tables(
                mp20_fieldforge_stats,
                ["MP20", MODEL],
                args.mp20_output_dir / "element_distribution_with_mp20",
                args.dpi,
            )
            print("Updated only the FieldForge and MP20 periodic-table plots.")
            return 0
        mp20_args = Namespace(
            output_dir=args.mp20_output_dir,
            model_element_cache_csv=args.cache_csv,
            mp20_element_cache_csv=args.mp20_element_cache_csv,
            model_order=model_order,
            include_model=[],
            dpi=args.dpi,
        )
        mp20_outputs.plot_mp20_element_outputs(mp20_args)
    else:
        print(f"Warning: MP20 element cache not found; skipped MP20 comparison: {args.mp20_element_cache_csv}")

    ok = sum(row["status"] == "ok" for row in new_rows)
    errors = len(new_rows) - ok
    print(f"Updated {MODEL}: {ok} valid CIF compositions, {errors} parsing errors")
    print(f"Wrote element outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
