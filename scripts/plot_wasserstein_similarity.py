#!/usr/bin/env python3
"""Compare generated models to MP20 with Wasserstein distances."""

from __future__ import annotations

import argparse
import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("CRYSTAL_BENCHMARK_OUTPUT_DIR", "outputs")) / ".mplconfig"),
)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from scipy.stats import wasserstein_distance

from settings import (
    CIF_MODELS,
    CRYSTALITE_NAME,
    FIELD_FORGE_NAME,
    MP20_CSV,
    OUTPUT_ROOT,
)

MODEL_REGISTRY = {name: Path(path) for name, path in CIF_MODELS}
DEFAULT_MODEL_ORDER = [FIELD_FORGE_NAME, CRYSTALITE_NAME]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    cif_dir: Path


def _safe_filename(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in name).strip("_") or "model"


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _axis_style(ax) -> None:
    ax.grid(axis="y", color="#dce7e8", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors="#263238", length=0)
    for spine in ax.spines.values():
        spine.set_color("#263238")
    ax.set_facecolor("#fbfdfd")


def _model_colors(model_order: Sequence[str]) -> dict[str, str]:
    base = {
        "MP20": "#30343b",
        FIELD_FORGE_NAME: "#ff7f50",
        CRYSTALITE_NAME: "#6c8ff6",
    }
    palette = list(mpl.colormaps["tab20"].colors)
    colors = {}
    for idx, model in enumerate(model_order):
        colors[model] = base.get(model, mpl.colors.to_hex(palette[idx % len(palette)]))
    return colors


def parse_structure_record(model: str, source: str, structure: Structure) -> dict[str, object]:
    return {
        "model": model,
        "source": source,
        "formula": structure.composition.reduced_formula,
        "density_g_cm3": float(structure.density),
        "n_unique_elements": int(len(structure.composition.elements)),
        "n_sites": int(len(structure)),
        "status": "ok",
        "error": "",
    }


def parse_cif_file(model: str, path: Path) -> dict[str, object]:
    try:
        structure = Structure.from_file(str(path), primitive=False, sort=False)
        return parse_structure_record(model, str(path), structure)
    except Exception as exc:
        return {
            "model": model,
            "source": str(path),
            "formula": "",
            "density_g_cm3": np.nan,
            "n_unique_elements": np.nan,
            "n_sites": np.nan,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def parse_mp20_row(row: pd.Series) -> dict[str, object]:
    try:
        structure = Structure.from_str(str(row["cif"]), fmt="cif", primitive=False, sort=False)
        record = parse_structure_record("MP20", str(row.get("material_id", row.name)), structure)
        elements = row.get("elements")
        if isinstance(elements, str) and elements:
            try:
                record["n_unique_elements"] = len(ast.literal_eval(elements))
            except Exception:
                pass
        return record
    except Exception as exc:
        return {
            "model": "MP20",
            "source": str(row.get("material_id", row.name)),
            "formula": str(row.get("pretty_formula", "")),
            "density_g_cm3": np.nan,
            "n_unique_elements": np.nan,
            "n_sites": np.nan,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_model_specs(models: Sequence[str], custom_models: Optional[Sequence[Sequence[str]]]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for name in models:
        if name not in MODEL_REGISTRY:
            raise SystemExit(f"Unknown model {name!r}. Registered models: {', '.join(MODEL_REGISTRY)}")
        specs.append(ModelSpec(name, MODEL_REGISTRY[name]))
    for item in custom_models or []:
        name, cif_dir = item
        specs.append(ModelSpec(str(name), Path(cif_dir).expanduser()))
    seen = set()
    deduped = []
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        if not spec.cif_dir.exists():
            raise SystemExit(f"CIF directory does not exist for {spec.name}: {spec.cif_dir}")
        deduped.append(spec)
    return deduped


def analyze_models(specs: Sequence[ModelSpec], limit: int) -> pd.DataFrame:
    rows = []
    for spec in specs:
        cif_files = sorted(spec.cif_dir.glob("*.cif"))[:limit]
        print(f"{spec.name}: parsing {len(cif_files)} CIF files from {spec.cif_dir}")
        rows.extend(parse_cif_file(spec.name, path) for path in cif_files)
    return pd.DataFrame(rows)


def analyze_mp20(mp20_csv: Path, limit: int) -> pd.DataFrame:
    nrows = None if limit <= 0 else limit
    print(f"MP20: parsing {'all' if nrows is None else nrows} CIF strings from {mp20_csv}")
    df = pd.read_csv(mp20_csv, usecols=["material_id", "pretty_formula", "elements", "cif"], nrows=nrows)
    return pd.DataFrame(parse_mp20_row(row) for _, row in df.iterrows())


def load_or_build_property_cache(args, specs: Sequence[ModelSpec]) -> pd.DataFrame:
    cache = args.cache_csv
    if args.use_cache and cache.exists():
        print(f"Using cached property values: {cache}")
        return pd.read_csv(cache)
    rows = [analyze_mp20(args.mp20_csv, args.mp20_limit), analyze_models(specs, args.limit)]
    values = pd.concat(rows, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    values.to_csv(cache, index=False)
    return values


def compute_wasserstein(values: pd.DataFrame, model_order: Sequence[str]) -> pd.DataFrame:
    ok = values[values["status"] == "ok"].copy()
    mp20_density = ok[ok["model"] == "MP20"]["density_g_cm3"].dropna().to_numpy()
    mp20_nel = ok[ok["model"] == "MP20"]["n_unique_elements"].dropna().to_numpy()
    records = []
    for model in model_order:
        subset = ok[ok["model"] == model]
        density = subset["density_g_cm3"].dropna().to_numpy()
        nel = subset["n_unique_elements"].dropna().to_numpy()
        records.append(
            {
                "model": model,
                "W_rho_density_g_cm3": wasserstein_distance(mp20_density, density) if len(density) else np.nan,
                "W_N_el": wasserstein_distance(mp20_nel, nel) if len(nel) else np.nan,
                "n_model_valid": int(len(subset)),
                "n_mp20_valid": int(len(mp20_density)),
                "model_density_mean": float(np.mean(density)) if len(density) else np.nan,
                "mp20_density_mean": float(np.mean(mp20_density)) if len(mp20_density) else np.nan,
                "model_n_unique_elements_mean": float(np.mean(nel)) if len(nel) else np.nan,
                "mp20_n_unique_elements_mean": float(np.mean(mp20_nel)) if len(mp20_nel) else np.nan,
            }
        )
    return pd.DataFrame(records)


def plot_distribution_comparison(values: pd.DataFrame, model_order: Sequence[str], out_dir: Path, dpi: int) -> None:
    ok = values[values["status"] == "ok"].copy()
    plot_order = ["MP20", *model_order]
    colors = _model_colors(plot_order)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), gridspec_kw={"wspace": 0.26})
    fig.patch.set_facecolor("#fbfdfd")

    ax = axes[0]
    bins = np.linspace(0, min(14, ok["density_g_cm3"].quantile(0.995)), 48)
    for model in plot_order:
        vals = ok[ok["model"] == model]["density_g_cm3"].dropna().to_numpy()
        ax.hist(vals, bins=bins, density=True, histtype="stepfilled", alpha=0.24, color=colors[model], edgecolor=colors[model], linewidth=1.2, label=model)
    ax.set_xlabel(r"Density $\rho$ (g cm$^{-3}$)", fontsize=12)
    ax.set_ylabel("Normalized density", fontsize=12)
    ax.set_title(r"Crystal Density Distribution", fontsize=14, fontweight="bold")
    _axis_style(ax)

    ax = axes[1]
    bins_labels = ["1", "2", "3", "4", "5", "6+"]
    x = np.arange(len(bins_labels), dtype=float)
    width = min(0.13, 0.84 / len(plot_order))
    offsets = (np.arange(len(plot_order)) - (len(plot_order) - 1) / 2.0) * width
    for offset, model in zip(offsets, plot_order):
        vals = ok[ok["model"] == model]["n_unique_elements"].dropna().astype(int)
        binned = vals.map(lambda v: "6+" if v >= 6 else str(v))
        counts = binned.value_counts()
        total = max(1, int(counts.sum()))
        pct = [100.0 * counts.get(label, 0) / total for label in bins_labels]
        ax.bar(x + offset, pct, width=width, color=colors[model], edgecolor="white", linewidth=0.6, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(bins_labels)
    ax.set_xlabel(r"Unique elements per material $N_{\mathrm{el}}$", fontsize=12)
    ax.set_ylabel("Normalized percentage", fontsize=12)
    ax.set_title("Compositional Complexity Distribution", fontsize=14, fontweight="bold")
    _axis_style(ax)
    ax.legend(frameon=True, facecolor="white", edgecolor="#d0d8dc", fontsize=8.5)

    fig.suptitle("Generated Models vs MP20 Reference Distributions", fontsize=17, fontweight="bold", y=0.98)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "property_distribution_comparison.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_wasserstein_bars(distances: pd.DataFrame, model_order: Sequence[str], out_dir: Path, dpi: int) -> None:
    colors = _model_colors(model_order)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), gridspec_kw={"wspace": 0.28})
    fig.patch.set_facecolor("#fbfdfd")
    metrics = [
        ("W_rho_density_g_cm3", r"$W_{\rho}$", r"Density Wasserstein distance (g cm$^{-3}$)"),
        ("W_N_el", r"$W_{N_{\mathrm{el}}}$", r"$N_{\mathrm{el}}$ Wasserstein distance"),
    ]
    for ax, (col, title, ylabel) in zip(axes, metrics):
        vals = [float(distances.set_index("model").loc[model, col]) for model in model_order]
        x = np.arange(len(model_order))
        ax.bar(x, vals, color=[colors[m] for m in model_order], edgecolor="white", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=28, ha="right", fontsize=10)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        _axis_style(ax)
    fig.suptitle("Wasserstein Distance to MP20", fontsize=17, fontweight="bold", y=0.98)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "wasserstein_distance_bars.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute W_rho and W_N_el to MP20 for generated CIF models.")
    parser.add_argument("--mp20-csv", type=Path, default=MP20_CSV)
    parser.add_argument("--mp20-limit", type=int, default=2500, help="Number of MP20 rows to parse; use <=0 for all rows.")
    parser.add_argument("--limit", type=int, default=2500, help="Number of CIFs per generated model.")
    parser.add_argument("--model", action="append", choices=sorted(MODEL_REGISTRY), default=None)
    parser.add_argument("--custom-model", action="append", nargs=2, metavar=("NAME", "CIF_DIR"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "wasserstein")
    parser.add_argument(
        "--cache-csv",
        type=Path,
        default=OUTPUT_ROOT / "wasserstein/csv/property_values.csv",
    )
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _configure_style()
    model_order = args.model or DEFAULT_MODEL_ORDER
    specs = build_model_specs(model_order, args.custom_model)
    model_order = [spec.name for spec in specs]
    values = load_or_build_property_cache(args, specs)
    distances = compute_wasserstein(values, model_order)
    csv_dir = args.output_dir / "csv"
    fig_dir = args.output_dir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    distances.to_csv(csv_dir / "wasserstein_distances_to_mp20.csv", index=False)
    values[values["status"] != "ok"].to_csv(csv_dir / "property_parse_errors.csv", index=False)
    plot_distribution_comparison(values, model_order, fig_dir, args.dpi)
    plot_wasserstein_bars(distances, model_order, fig_dir, args.dpi)
    print(distances.to_string(index=False))
    print(f"Wrote Wasserstein outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
