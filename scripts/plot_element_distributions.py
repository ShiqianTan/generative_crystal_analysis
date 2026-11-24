#!/usr/bin/env python3
"""Analyze and plot element distributions for generated CIF datasets.

This script combines pymatgen parsing with pymatviz periodic-table heatmaps.
It writes per-model element statistics, a top-element comparison, and one
periodic-table frequency plot per model.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("CRYSTAL_BENCHMARK_OUTPUT_DIR", "outputs")) / ".mplconfig"),
)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymatviz as pmv
from pymatgen.core import Composition, Element, Structure

from settings import CIF_MODELS, OUTPUT_ROOT

MAIN_MODELS = CIF_MODELS
MODEL_PRESETS = {"main": MAIN_MODELS}
DEFAULT_MODELS = MAIN_MODELS

ELEMENTS = [Element.from_Z(z).symbol for z in range(1, 119)]


@dataclass(frozen=True)
class ModelInput:
    name: str
    path: Path


def _safe_filename(name: str) -> str:
    safe = []
    for char in name:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "model"


def _parse_model_specs(specs: Optional[Sequence[Sequence[str]]], preset: str = "main") -> List[ModelInput]:
    raw_specs = specs or MODEL_PRESETS[preset]
    models = [ModelInput(str(name), Path(path).expanduser()) for name, path in raw_specs]
    seen = set()
    for model in models:
        if model.name in seen:
            raise SystemExit(f"Duplicate model name: {model.name}")
        seen.add(model.name)
        if not model.path.exists():
            raise SystemExit(f"Path does not exist for {model.name}: {model.path}")
        if not model.path.is_dir():
            raise SystemExit(f"Path is not a directory for {model.name}: {model.path}")
    return models


def _find_cifs(path: Path, recursive: bool, limit: Optional[int]) -> List[Path]:
    pattern = "**/*.cif" if recursive else "*.cif"
    cifs = sorted(path.glob(pattern))
    if limit is not None:
        cifs = cifs[:limit]
    return cifs


def _analyze_one(job: Tuple[str, str]) -> Dict[str, object]:
    model, cif_path = job
    path = Path(cif_path)
    try:
        structure = Structure.from_file(str(path), primitive=False, sort=False)
        comp = structure.composition
        elem_amounts = {el.symbol: float(amount) for el, amount in comp.items()}
        return {
            "model": model,
            "file": str(path),
            "formula": comp.reduced_formula,
            "n_sites": int(len(structure)),
            "element_amounts": elem_amounts,
            "status": "ok",
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - keep the batch moving.
        return {
            "model": model,
            "file": str(path),
            "formula": "",
            "n_sites": "",
            "element_amounts": {},
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def analyze_models(
    models: Sequence[ModelInput],
    recursive: bool,
    limit: Optional[int],
    workers: int,
) -> List[Dict[str, object]]:
    jobs = []
    for model in models:
        cifs = _find_cifs(model.path, recursive=recursive, limit=limit)
        if not cifs:
            raise SystemExit(f"No CIF files found for {model.name}: {model.path}")
        jobs.extend((model.name, str(cif)) for cif in cifs)

    total = len(jobs)
    rows: List[Dict[str, object]] = []
    print(f"Analyzing element compositions for {total} CIF files...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_analyze_one, job) for job in jobs]
        for idx, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if idx % 250 == 0 or idx == total:
                print(f"  {idx}/{total} files processed", flush=True)
    return rows


def write_structure_cache(rows: Sequence[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "file", "formula", "n_sites", "status", "error", "element_amounts"]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            amounts = row["element_amounts"]
            writer.writerow(
                {
                    **{key: row[key] for key in fields if key != "element_amounts"},
                    "element_amounts": ";".join(
                        f"{elem}:{amount:g}" for elem, amount in sorted(amounts.items())
                    ),
                }
            )


def read_structure_cache(input_csv: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with input_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            amounts = {}
            if row["element_amounts"]:
                for item in row["element_amounts"].split(";"):
                    elem, amount = item.split(":", 1)
                    amounts[elem] = float(amount)
            rows.append({**row, "element_amounts": amounts})
    return rows


def _element_stats_for_model(
    rows: Sequence[Dict[str, object]], model: str
) -> pd.DataFrame:
    model_rows = [row for row in rows if row["model"] == model and row["status"] == "ok"]
    compositions = [row["element_amounts"] for row in model_rows]
    if compositions:
        atom_counts = pmv.count_elements(
            [Composition(comp) for comp in compositions], fill_value=0
        )
    else:
        atom_counts = pd.Series(0.0, index=ELEMENTS)
    atom_counts = atom_counts.reindex(ELEMENTS, fill_value=0).astype(float)

    presence = Counter()
    for comp in compositions:
        for elem, amount in comp.items():
            if amount > 0:
                presence[elem] += 1

    total_atoms = float(atom_counts.sum())
    total_structures = len(model_rows)
    data = []
    for elem in ELEMENTS:
        count = float(atom_counts.get(elem, 0.0))
        structures_with_element = int(presence.get(elem, 0))
        data.append(
            {
                "model": model,
                "element": elem,
                "atomic_number": Element(elem).Z,
                "atom_count": count,
                "atom_fraction": count / total_atoms if total_atoms else 0.0,
                "atom_percent": 100.0 * count / total_atoms if total_atoms else 0.0,
                "structures_with_element": structures_with_element,
                "structure_fraction": (
                    structures_with_element / total_structures if total_structures else 0.0
                ),
                "structure_percent": (
                    100.0 * structures_with_element / total_structures
                    if total_structures
                    else 0.0
                ),
                "n_structures": total_structures,
                "total_atoms": total_atoms,
            }
        )
    return pd.DataFrame(data)


def build_element_stats(
    rows: Sequence[Dict[str, object]], model_order: Sequence[str]
) -> pd.DataFrame:
    return pd.concat(
        [_element_stats_for_model(rows, model) for model in model_order],
        ignore_index=True,
    )


def write_element_stats(
    stats: pd.DataFrame, model_order: Sequence[str], output_dir: Path
) -> None:
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(csv_dir / "combined_element_distribution.csv", index=False)
    for model in model_order:
        model_df = stats[stats["model"] == model].copy()
        model_df.to_csv(csv_dir / f"{_safe_filename(model)}_element_distribution.csv", index=False)


def _model_colors(model_order: Sequence[str]) -> Dict[str, Tuple[float, float, float, float]]:
    palette = [
        "#22a6d5",
        "#ff8f3d",
        "#50c878",
        "#b06cd5",
        "#e05d7b",
        "#7aa6ff",
        "#f2c94c",
        "#00b8a9",
        "#ef6f6c",
        "#6c8ebf",
    ]
    return {
        model: mpl.colors.to_rgba(palette[idx % len(palette)])
        for idx, model in enumerate(model_order)
    }


def plot_top_element_comparison(
    stats: pd.DataFrame,
    model_order: Sequence[str],
    output_dir: Path,
    top_n: int,
    dpi: int,
) -> None:
    plot_dir = output_dir / "comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)
    total_by_element = (
        stats.groupby("element", as_index=False)["atom_count"].sum()
        .sort_values("atom_count", ascending=False)
        .head(top_n)
    )
    top_elements = total_by_element["element"].tolist()
    comparison = stats[stats["element"].isin(top_elements)].copy()
    comparison["element"] = pd.Categorical(
        comparison["element"], categories=top_elements, ordered=True
    )
    comparison["model"] = pd.Categorical(comparison["model"], categories=model_order, ordered=True)
    comparison = comparison.sort_values(["element", "model"])
    comparison.to_csv(plot_dir / f"top{top_n}_element_comparison.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    model_colors = _model_colors(model_order)
    y = np.arange(len(top_elements), dtype=float)
    height = min(0.11, 0.82 / max(1, len(model_order)))
    offsets = (np.arange(len(model_order)) - (len(model_order) - 1) / 2.0) * height

    fig_height = max(6.2, 0.32 * len(top_elements) + 2.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    fig.set_layout_engine(None)
    fig.patch.set_facecolor("#fbfdfd")
    fig.suptitle(f"Top {top_n} Element Distribution Comparison", fontsize=17, fontweight="bold", y=0.985)

    for offset, model in zip(offsets, model_order):
        model_df = stats[stats["model"] == model].set_index("element")
        values = [float(model_df.loc[elem, "atom_percent"]) for elem in top_elements]
        ax.barh(
            y + offset,
            values,
            height=height,
            label=model,
            color=model_colors[model],
            edgecolor="white",
            linewidth=0.65,
            alpha=0.96,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(top_elements, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Atomic fraction (%)", fontsize=11, color="#37474f")
    ax.grid(axis="x", color="#dce7e8", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors="#37474f", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd7da")
    ax.spines["bottom"].set_color("#cbd7da")
    ax.set_facecolor("#fbfdfd")

    handles = [
        mpl.patches.Patch(facecolor=model_colors[model], edgecolor="white", label=model)
        for model in model_order
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=min(4, len(model_order)),
        frameon=False,
        fontsize=10,
    )
    ax.set_position([0.10, 0.08, 0.88, 0.66])

    output = plot_dir / f"top{top_n}_element_comparison.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_periodic_tables(
    stats: pd.DataFrame,
    model_order: Sequence[str],
    output_dir: Path,
    dpi: int,
) -> None:
    ptable_dir = output_dir / "periodic_tables"
    ptable_dir.mkdir(parents=True, exist_ok=True)

    for model in model_order:
        model_df = stats[stats["model"] == model]
        data = pd.Series(
            model_df["atom_percent"].to_numpy(),
            index=model_df["element"].to_numpy(),
            name="atom_percent",
        )
        fig = pmv.ptable_heatmap(
            data,
            colormap="YlGnBu",
            value_show_mode="value",
            value_fmt=".1f",
            cbar_title="Atomic fraction (%)",
            return_type="figure",
            hide_f_block=False,
        )
        fig.set_size_inches(12.5, 6.4)
        fig.patch.set_facecolor("#fbfdfd")
        output = ptable_dir / f"{_safe_filename(model)}_periodic_table_atom_percent.png"
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Wrote periodic table heatmap: {output}")


def unique_element_distribution(
    rows: Sequence[Dict[str, object]], model_order: Sequence[str]
) -> pd.DataFrame:
    bins = ["1", "2", "3", "4", "5", "6+"]
    records = []
    for model in model_order:
        model_rows = [row for row in rows if row["model"] == model and row["status"] == "ok"]
        counts = Counter()
        for row in model_rows:
            n_unique = sum(1 for amount in row["element_amounts"].values() if amount > 0)
            label = "6+" if n_unique >= 6 else str(n_unique)
            counts[label] += 1
        total = len(model_rows)
        for label in bins:
            count = counts.get(label, 0)
            records.append(
                {
                    "model": model,
                    "unique_elements_per_material": label,
                    "count": count,
                    "normalized_percentage": 100.0 * count / total if total else 0.0,
                    "n_structures": total,
                }
            )
    return pd.DataFrame(records)


def plot_unique_elements_distribution(
    rows: Sequence[Dict[str, object]],
    model_order: Sequence[str],
    output_dir: Path,
    dpi: int,
) -> None:
    plot_dir = output_dir / "unique_elements_per_material"
    plot_dir.mkdir(parents=True, exist_ok=True)
    dist = unique_element_distribution(rows, model_order)
    dist.to_csv(plot_dir / "unique_elements_per_material_distribution.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    model_colors = _model_colors(model_order)
    bins = ["1", "2", "3", "4", "5", "6+"]
    x = np.arange(len(bins), dtype=float)
    width = min(0.11, 0.82 / max(1, len(model_order)))
    offsets = (np.arange(len(model_order)) - (len(model_order) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    fig.patch.set_facecolor("#fbfdfd")
    for offset, model in zip(offsets, model_order):
        model_df = dist[dist["model"] == model].set_index("unique_elements_per_material")
        values = [float(model_df.loc[label, "normalized_percentage"]) for label in bins]
        ax.bar(
            x + offset,
            values,
            width=width,
            label=model,
            color=model_colors[model],
            edgecolor="white",
            linewidth=0.65,
            alpha=0.96,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=13)
    ax.set_xlabel("Unique Elements per Material", fontsize=16)
    ax.set_ylabel("Normalized Percentage", fontsize=16)
    ax.grid(axis="y", color="#dce7e8", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors="#1f2d33", length=0)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_color("#1f2d33")
    ax.spines["bottom"].set_color("#1f2d33")
    ax.spines["top"].set_color("#1f2d33")
    ax.spines["right"].set_color("#1f2d33")
    ax.set_facecolor("#fbfdfd")
    ymax = max(1.0, float(dist["normalized_percentage"].max()))
    ax.set_ylim(0, min(100.0, ymax * 1.18 + 2.0))
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=10,
        ncol=1 if len(model_order) <= 5 else 2,
    )
    fig.tight_layout()

    output = plot_dir / "unique_elements_per_material_distribution.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote unique-elements distribution: {output}")


def summarize(rows: Sequence[Dict[str, object]], model_order: Sequence[str]) -> None:
    print("\nSummary")
    for model in model_order:
        model_rows = [row for row in rows if row["model"] == model]
        ok = sum(1 for row in model_rows if row["status"] == "ok")
        errors = len(model_rows) - ok
        total_atoms = sum(
            sum(row["element_amounts"].values())
            for row in model_rows
            if row["status"] == "ok"
        )
        print(f"  {model}: {ok} structures, {total_atoms:g} atoms, {errors} errors")
    first_error = next((row for row in rows if row["status"] == "error"), None)
    if first_error:
        print(f"  First error: {first_error['file']} -> {first_error['error']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze CIF element distributions and plot pymatviz periodic tables."
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("NAME", "CIF_DIR"),
        help="Model name and CIF directory. Repeat for multiple models.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(MODEL_PRESETS),
        default="main",
        help="Built-in model list used when --model is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / "elements",
        help="Output directory for CSV files and plots.",
    )
    parser.add_argument(
        "--cache-csv",
        type=Path,
        default=OUTPUT_ROOT / "elements/csv/structure_element_cache.csv",
        help="Per-structure composition cache CSV.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Read structure compositions from --cache-csv instead of reparsing CIFs.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search CIF files recursively under each model directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2500,
        help="Maximum CIF files per model, taken after lexical sorting.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
        help="Number of parser worker threads.",
    )
    parser.add_argument(
        "--top-elements",
        type=int,
        default=20,
        help="Number of globally most frequent elements to compare across models.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Raster output DPI.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    models = _parse_model_specs(args.model, args.preset)
    model_order = [model.name for model in models]

    if args.use_cache:
        if not args.cache_csv.exists():
            raise SystemExit(f"Cache file not found: {args.cache_csv}")
        rows = read_structure_cache(args.cache_csv)
    else:
        rows = analyze_models(
            models=models,
            recursive=args.recursive,
            limit=args.limit,
            workers=args.workers,
        )
        write_structure_cache(rows, args.cache_csv)
        print(f"Wrote structure composition cache: {args.cache_csv}")

    summarize(rows, model_order)
    stats = build_element_stats(rows, model_order)
    write_element_stats(stats, model_order, args.output_dir)
    plot_top_element_comparison(stats, model_order, args.output_dir, args.top_elements, args.dpi)
    plot_unique_elements_distribution(rows, model_order, args.output_dir, args.dpi)
    plot_periodic_tables(stats, model_order, args.output_dir, args.dpi)
    print(f"Wrote element distribution outputs under: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
