#!/usr/bin/env python3
"""Add MP20 reference distributions to existing model-comparison outputs."""

from __future__ import annotations

import argparse
import ast
import csv
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("CRYSTAL_BENCHMARK_OUTPUT_DIR", "outputs")) / ".mplconfig"),
)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import plot_crystal_symmetry_distributions as sgplot
import plot_element_distributions as elplot
from settings import MP20_CSV, OUTPUT_ROOT


MODEL_ORDER_MAIN = [name for name, _ in sgplot.MAIN_MODELS]
MODEL_ORDER_WITH_MP20 = ["MP20", *MODEL_ORDER_MAIN]
REDUCED_MODEL_ORDER = list(MODEL_ORDER_MAIN)
ELEMENTS = [Element.from_Z(z).symbol for z in range(1, 119)]


def _safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)


def _frac_close_to_integer(values: np.ndarray, tol: float) -> bool:
    return bool(np.all(np.abs(values - np.round(values)) <= tol))


def _wyckoff_dimension(frac_coord, symm_ops, tol: float = 1e-5) -> int:
    coord = np.asarray(frac_coord, dtype=float)
    constraints = []
    eye = np.eye(3)
    for op in symm_ops:
        rot = np.asarray(op.rotation_matrix, dtype=float)
        trans = np.asarray(op.translation_vector, dtype=float)
        diff = rot.dot(coord) + trans - coord
        if _frac_close_to_integer(diff, tol):
            constraints.append(rot - eye)
    if not constraints:
        return 3
    rank = int(np.linalg.matrix_rank(np.vstack(constraints), tol=1e-7))
    return max(0, min(3, 3 - rank))


def _dataset_value(dataset, key: str):
    return getattr(dataset, key) if hasattr(dataset, key) else dataset[key]


def _parse_elements(elements_text: str) -> List[str]:
    try:
        values = ast.literal_eval(elements_text)
        return [str(value) for value in values]
    except Exception:
        return []


def analyze_mp20(mp20_csv: Path, symmetry_csv: Path, element_cache_csv: Path, symprec: float) -> None:
    symmetry_csv.parent.mkdir(parents=True, exist_ok=True)
    element_cache_csv.parent.mkdir(parents=True, exist_ok=True)

    sg_fields = [
        "record_type",
        "model",
        "file",
        "spacegroup_number",
        "spacegroup_symbol",
        "hall_symbol",
        "wyckoff_letter",
        "multiplicity",
        "dimensionality",
        "status",
        "error",
    ]
    elem_fields = ["model", "file", "formula", "n_sites", "status", "error", "element_amounts"]

    with mp20_csv.open(newline="", encoding="utf-8") as in_handle, symmetry_csv.open(
        "w", newline="", encoding="utf-8"
    ) as sg_handle, element_cache_csv.open("w", newline="", encoding="utf-8") as elem_handle:
        reader = csv.DictReader(in_handle)
        sg_writer = csv.DictWriter(sg_handle, fieldnames=sg_fields)
        elem_writer = csv.DictWriter(elem_handle, fieldnames=elem_fields)
        sg_writer.writeheader()
        elem_writer.writeheader()

        for idx, row in enumerate(reader):
            material_id = row.get("material_id") or f"mp20_{idx}"
            source_id = f"MP20:{material_id}"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    structure = Structure.from_str(row["cif"], fmt="cif")
                comp = structure.composition
                amounts = {el.symbol: float(amount) for el, amount in comp.items()}
                elem_writer.writerow(
                    {
                        "model": "MP20",
                        "file": source_id,
                        "formula": row.get("pretty_formula") or comp.reduced_formula,
                        "n_sites": len(structure),
                        "status": "ok",
                        "error": "",
                        "element_amounts": ";".join(
                            f"{elem}:{amount:g}" for elem, amount in sorted(amounts.items())
                        ),
                    }
                )

                sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=5.0)
                dataset = sga.get_symmetry_dataset()
                symm_ops = sga.get_space_group_operations()
                sg_number = int(_dataset_value(dataset, "number"))
                international = str(_dataset_value(dataset, "international"))
                hall = str(_dataset_value(dataset, "hall"))
                wyckoffs = list(_dataset_value(dataset, "wyckoffs"))
                equivalent_atoms = np.asarray(_dataset_value(dataset, "equivalent_atoms"))
                sg_writer.writerow(
                    {
                        "record_type": "structure",
                        "model": "MP20",
                        "file": source_id,
                        "spacegroup_number": sg_number,
                        "spacegroup_symbol": international,
                        "hall_symbol": hall,
                        "wyckoff_letter": "",
                        "multiplicity": "",
                        "dimensionality": "",
                        "status": "ok",
                        "error": "",
                    }
                )
                for eq_id in sorted(set(int(x) for x in equivalent_atoms)):
                    indices = np.where(equivalent_atoms == eq_id)[0]
                    rep_index = int(indices[0])
                    dim = _wyckoff_dimension(structure[rep_index].frac_coords, symm_ops)
                    sg_writer.writerow(
                        {
                            "record_type": "wyckoff",
                            "model": "MP20",
                            "file": source_id,
                            "spacegroup_number": sg_number,
                            "spacegroup_symbol": international,
                            "hall_symbol": hall,
                            "wyckoff_letter": str(wyckoffs[rep_index]),
                            "multiplicity": len(indices),
                            "dimensionality": dim,
                            "status": "ok",
                            "error": "",
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                elem_symbols = _parse_elements(row.get("elements", ""))
                comp = Composition(row.get("pretty_formula", "")) if row.get("pretty_formula") else None
                amounts = {el.symbol: float(amount) for el, amount in comp.items()} if comp else {
                    elem: 1.0 for elem in elem_symbols
                }
                elem_writer.writerow(
                    {
                        "model": "MP20",
                        "file": source_id,
                        "formula": row.get("pretty_formula", ""),
                        "n_sites": "",
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "element_amounts": ";".join(
                            f"{elem}:{amount:g}" for elem, amount in sorted(amounts.items())
                        ),
                    }
                )
                sg_writer.writerow(
                    {
                        "record_type": "error",
                        "model": "MP20",
                        "file": source_id,
                        "spacegroup_number": row.get("spacegroup_number", ""),
                        "spacegroup_symbol": "",
                        "hall_symbol": "",
                        "wyckoff_letter": "",
                        "multiplicity": "",
                        "dimensionality": "",
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if (idx + 1) % 2500 == 0:
                print(f"  MP20 parsed {idx + 1} rows", flush=True)


def load_or_build_mp20(args) -> None:
    if args.rebuild_mp20 or not args.mp20_symmetry_csv.exists() or not args.mp20_element_cache_csv.exists():
        print("Building MP20 symmetry and element caches...")
        analyze_mp20(args.mp20_csv, args.mp20_symmetry_csv, args.mp20_element_cache_csv, args.symprec)
    else:
        print("Using existing MP20 caches.")


def combine_csv_rows(paths: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        rows.extend(sgplot.read_csv(path))
    return rows


def build_element_rows_from_cache(cache_csv: Path) -> List[Dict[str, object]]:
    return elplot.read_structure_cache(cache_csv)


def _filter_rows_by_model(rows: Sequence[Dict[str, object]], model_order: Sequence[str]) -> List[Dict[str, object]]:
    allowed = set(model_order)
    return [row for row in rows if row.get("model") in allowed]


def _resolve_model_order(args, rows: Sequence[Dict[str, object]]) -> List[str]:
    order = list(args.model_order) if args.model_order else list(MODEL_ORDER_MAIN)
    for model in args.include_model or []:
        if model not in order:
            order.append(model)
    return _validate_model_order(order, rows)


def _validate_model_order(order: Sequence[str], rows: Sequence[Dict[str, object]]) -> List[str]:
    order = list(order)
    available = {str(row.get("model")) for row in rows}
    missing = [model for model in order if model not in available]
    if missing:
        raise SystemExit(
            "Requested model(s) not found in the supplied cache: "
            + ", ".join(missing)
            + ". Regenerate the cache with those models or pass a cache that contains them."
        )
    return order


def plot_mp20_element_outputs(args) -> None:
    output_dir = args.output_dir / "element_distribution_with_mp20"
    raw_model_rows = build_element_rows_from_cache(args.model_element_cache_csv)
    model_order = _resolve_model_order(args, raw_model_rows)
    model_rows = _filter_rows_by_model(raw_model_rows, model_order)
    mp20_rows = build_element_rows_from_cache(args.mp20_element_cache_csv)
    all_rows = mp20_rows + model_rows
    model_order_with_mp20 = ["MP20", *model_order]

    stats = elplot.build_element_stats(all_rows, model_order_with_mp20)
    elplot.write_element_stats(stats, model_order_with_mp20, output_dir)
    elplot.plot_top_element_comparison(stats, model_order_with_mp20, output_dir, 20, args.dpi)
    elplot.plot_periodic_tables(stats, model_order_with_mp20, output_dir, args.dpi)
    elplot.plot_unique_elements_distribution(all_rows, model_order_with_mp20, output_dir, args.dpi)

    mp20_stats = elplot.build_element_stats(mp20_rows, ["MP20"])
    mp20_dir = output_dir / "mp20_only"
    elplot.write_element_stats(mp20_stats, ["MP20"], mp20_dir)
    elplot.plot_periodic_tables(mp20_stats, ["MP20"], mp20_dir, args.dpi)
    elplot.plot_unique_elements_distribution(mp20_rows, ["MP20"], mp20_dir, args.dpi)


def plot_mp20_symmetry_outputs(args) -> None:
    output_dir = args.output_dir / "symmetry_with_mp20"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_model_rows = combine_csv_rows([args.model_symmetry_csv])
    model_order = _resolve_model_order(args, raw_model_rows)
    model_rows = _filter_rows_by_model(raw_model_rows, model_order)
    all_rows = combine_csv_rows([args.mp20_symmetry_csv]) + model_rows
    mp20_rows = combine_csv_rows([args.mp20_symmetry_csv])
    model_order_with_mp20 = ["MP20", *model_order]

    sgplot.write_csv(all_rows, output_dir / "combined_models_with_mp20_symmetry_records.csv")
    sgplot.write_csv(mp20_rows, output_dir / "mp20_symmetry_records.csv")
    sgplot.write_spacegroup_mapping_excel(
        all_rows,
        model_order_with_mp20,
        output_dir / "combined_models_with_mp20_spacegroup_mapping.xlsx",
    )
    sgplot.write_spacegroup_mapping_excel(
        mp20_rows,
        ["MP20"],
        output_dir / "mp20_spacegroup_mapping.xlsx",
    )

    sgplot.plot_figure(
        all_rows,
        model_order_with_mp20,
        output_dir / "combined_models_with_mp20_symmetry_distributions.png",
        max_spacegroups=32,
        title="Crystallographic Symmetry Distributions with MP20",
        dpi=args.dpi,
    )
    sgplot.plot_grouped_bar_figure(
        all_rows,
        model_order_with_mp20,
        output_dir / "combined_models_with_mp20_grouped_bars_top15.png",
        top_spacegroups=15,
        title="Grouped Comparison with MP20 Top Spacegroups",
        dpi=args.dpi,
        reference_model="MP20",
    )
    sgplot.plot_spacegroup_comparison_horizontal(
        all_rows,
        model_order_with_mp20,
        output_dir / "combined_models_with_mp20_spacegroup_horizontal_top15.png",
        top_spacegroups=15,
        title="Top 15 MP20 Spacegroup Comparison",
        dpi=args.dpi,
        reference_model="MP20",
        legend_mode="right",
        close_top_right=True,
    )

    reduced_model_order = _validate_model_order(
        args.reduced_model_order or REDUCED_MODEL_ORDER,
        raw_model_rows,
    )
    reduced_order_with_mp20 = ["MP20", *reduced_model_order]
    reduced_rows = _filter_rows_by_model(all_rows, reduced_order_with_mp20)
    reduced_output_dir = output_dir / "reduced_models"
    reduced_output_dir.mkdir(parents=True, exist_ok=True)
    sgplot.write_csv(
        reduced_rows,
        reduced_output_dir / "combined_models_with_mp20_symmetry_records.csv",
    )
    sgplot.write_spacegroup_mapping_excel(
        reduced_rows,
        reduced_order_with_mp20,
        reduced_output_dir / "combined_models_with_mp20_spacegroup_mapping.xlsx",
    )
    sgplot.plot_figure(
        reduced_rows,
        reduced_order_with_mp20,
        reduced_output_dir / "combined_models_with_mp20_symmetry_distributions.png",
        max_spacegroups=32,
        title="Reduced Crystallographic Symmetry Distributions with MP20",
        dpi=args.dpi,
    )
    sgplot.plot_grouped_bar_figure(
        reduced_rows,
        reduced_order_with_mp20,
        reduced_output_dir / "combined_models_with_mp20_grouped_bars_top15.png",
        top_spacegroups=15,
        title="Reduced Grouped Comparison with MP20 Top Spacegroups",
        dpi=args.dpi,
        reference_model="MP20",
    )
    sgplot.plot_spacegroup_comparison_horizontal(
        reduced_rows,
        reduced_order_with_mp20,
        reduced_output_dir / "combined_models_with_mp20_spacegroup_horizontal_top15.png",
        top_spacegroups=15,
        title="Reduced Top 15 MP20 Spacegroup Comparison",
        dpi=args.dpi,
        reference_model="MP20",
        legend_mode="right",
        close_top_right=True,
    )
    sgplot.plot_figure(
        mp20_rows,
        ["MP20"],
        output_dir / "mp20_symmetry_distributions.png",
        max_spacegroups=32,
        title="MP20 Crystallographic Symmetry Distributions",
        dpi=args.dpi,
    )
    sgplot.plot_grouped_bar_figure(
        mp20_rows,
        ["MP20"],
        output_dir / "mp20_grouped_bars_top15.png",
        top_spacegroups=15,
        title="MP20 Symmetry Properties",
        dpi=args.dpi,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add MP20 reference data to existing plots.")
    parser.add_argument("--mp20-csv", type=Path, default=MP20_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "mp20")
    parser.add_argument(
        "--mp20-symmetry-csv",
        type=Path,
        default=OUTPUT_ROOT / "mp20/csv/mp20_symmetry_records.csv",
    )
    parser.add_argument(
        "--mp20-element-cache-csv",
        type=Path,
        default=OUTPUT_ROOT / "mp20/csv/mp20_structure_element_cache.csv",
    )
    parser.add_argument(
        "--model-symmetry-csv",
        type=Path,
        default=OUTPUT_ROOT / "symmetry/csv/symmetry_records.csv",
    )
    parser.add_argument(
        "--model-element-cache-csv",
        type=Path,
        default=OUTPUT_ROOT / "elements/csv/structure_element_cache.csv",
    )
    parser.add_argument(
        "--model-order",
        action="append",
        default=[],
        help="Use exactly this model order from the supplied caches. Repeat for multiple models.",
    )
    parser.add_argument(
        "--include-model",
        action="append",
        default=[],
        help="Append one model from the supplied caches to the default order.",
    )
    parser.add_argument(
        "--reduced-model-order",
        action="append",
        default=[],
        help=(
            "Model order for the additional reduced MP20 symmetry comparison. "
            "Repeat for multiple models; default: "
            + ", ".join(REDUCED_MODEL_ORDER)
        ),
    )
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--rebuild-mp20", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    load_or_build_mp20(args)
    plot_mp20_symmetry_outputs(args)
    plot_mp20_element_outputs(args)
    print(f"Wrote MP20 reference outputs under: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
