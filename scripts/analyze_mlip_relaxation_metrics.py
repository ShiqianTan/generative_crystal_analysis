#!/usr/bin/env python3
"""Analyze multi-MLIP relaxation metrics from benchmark JSON files."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from collections import Counter
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

from settings import MLIP_JSON_DIR, MLIP_MODEL_SPECS, MP20_CSV, OUTPUT_ROOT

BASE_DIR = MLIP_JSON_DIR
ELEMENT_CACHE = OUTPUT_ROOT / "elements/csv/structure_element_cache.csv"
ALL_MAIN_MODEL_SPECS = MLIP_MODEL_SPECS
DEFAULT_MAIN_MODEL_NAMES = [spec[0] for spec in MLIP_MODEL_SPECS]
MAIN_MODEL_SPECS = ALL_MAIN_MODEL_SPECS

MODEL_REGISTRY = {
    name: (name, pattern, cif_dir)
    for name, pattern, cif_dir in ALL_MAIN_MODEL_SPECS
}
MODEL_PRESETS = {"main": MAIN_MODEL_SPECS, "all": ALL_MAIN_MODEL_SPECS}
MODEL_SPECS = MAIN_MODEL_SPECS

FALLBACK_PATTERNS = {}

METRICS = {
    "e_above_hull": {
        "section": "stability",
        "mean_key": "mean_e_above_hull",
        "x_label": r"$E_{\mathrm{hull}}$ (eV/atom)",
        "title": r"$E_{\mathrm{hull}}$",
        "bins": np.linspace(-0.1, 0.35, 55),
    },
    "relaxation_RMSD": {
        "section": "relaxation_stability",
        "mean_key": "mean_relaxation_RMSE",
        "x_label": r"Relaxation RMSD ($\AA$)",
        "title": "Relaxation RMSD",
        "bins": np.linspace(0, 1.0, 55),
    },
    "formation_energy": {
        "section": "formation_energy",
        "mean_key": "mean_formation_energy",
        "x_label": "Formation energy (eV/atom)",
        "title": "Formation energy",
        "bins": 55,
    },
}

PREDICTORS = [("mean", "value"), ("orb", "value_orb"), ("mace", "value_mace"), ("uma", "value_uma")]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    pattern: str
    cif_dir: Path


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


def _model_colors(model_order: Sequence[str]) -> Dict[str, Tuple[float, float, float, float]]:
    palette = []
    for cmap_name in ["tab20", "Set3", "Paired", "Dark2", "Accent"]:
        cmap = mpl.colormaps[cmap_name]
        size = getattr(cmap, "N", 12)
        palette.extend(mpl.colors.to_hex(cmap(i / max(1, size - 1))) for i in range(size))
    if len(model_order) > len(palette):
        palette.extend(mpl.colors.to_hex(c) for c in mpl.colormaps["turbo"](np.linspace(0.05, 0.95, len(model_order) - len(palette))))
    return {model: mpl.colors.to_rgba(palette[idx % len(palette)]) for idx, model in enumerate(model_order)}


def _predictor_colors() -> Dict[str, str]:
    return {"mean": "#1f77b4", "orb": "#ff7f0e", "mace": "#2ca02c", "uma": "#9467bd"}


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


def _dedupe_specs(specs: Sequence[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    seen = set()
    deduped = []
    for spec in specs:
        if spec[0] in seen:
            continue
        seen.add(spec[0])
        deduped.append(spec)
    return deduped


def load_model_specs(
    preset: str = "main",
    only_models: Optional[Sequence[str]] = None,
    include_models: Optional[Sequence[str]] = None,
    custom_models: Optional[Sequence[Sequence[str]]] = None,
) -> List[ModelSpec]:
    specs = []
    if only_models:
        for name in only_models:
            if name not in MODEL_REGISTRY:
                available = ", ".join(MODEL_REGISTRY)
                raise SystemExit(f"Unknown --only-model {name!r}. Available: {available}")
            specs.append(MODEL_REGISTRY[name])
    else:
        specs = list(MODEL_PRESETS[preset])
    for name in include_models or []:
        if name not in MODEL_REGISTRY:
            available = ", ".join(MODEL_REGISTRY)
            raise SystemExit(f"Unknown --include-model {name!r}. Available: {available}")
        specs.append(MODEL_REGISTRY[name])
    for item in custom_models or []:
        name, pattern, cif_dir = item
        specs.append((str(name), str(pattern), str(cif_dir)))
    return [ModelSpec(name, pattern, Path(cif_dir)) for name, pattern, cif_dir in _dedupe_specs(specs)]


def find_json_files(base_dir: Path, spec: ModelSpec, recursive: bool = True) -> Tuple[List[Path], str]:
    files = sorted(base_dir.glob(spec.pattern))
    pattern_used = spec.pattern
    if not files and recursive:
        files = sorted(base_dir.glob(f"**/{spec.pattern}"))
    if not files and spec.name in FALLBACK_PATTERNS:
        pattern_used = FALLBACK_PATTERNS[spec.name]
        files = sorted(base_dir.glob(pattern_used))
        if not files and recursive:
            files = sorted(base_dir.glob(f"**/{pattern_used}"))
    if not files:
        raise SystemExit(f"No JSON files found for {spec.name}: {spec.pattern}")
    return files, pattern_used


def _extract_number(text: str, key: str) -> float:
    pattern = rf"'{re.escape(key)}':\s*(?:np\.float64\()?([-+0-9.eE]+)"
    match = re.search(pattern, text)
    return float(match.group(1)) if match else float("nan")


def _extract_bracketed_list(text: str, start: int) -> str:
    list_start = text.find("[", start)
    if list_start < 0:
        raise ValueError("Could not find individual_values list")
    depth = 0
    for idx in range(list_start, len(text)):
        char = text[idx]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[list_start : idx + 1]
    raise ValueError("Unclosed individual_values list")


def _extract_individual_values(stability_text: str, section: str) -> List[Dict[str, float]]:
    section_start = stability_text.find(f"'{section}':")
    if section_start < 0:
        raise ValueError(f"Section not found: {section}")
    values_start = stability_text.find("individual_values=", section_start)
    if values_start < 0:
        raise ValueError(f"individual_values not found for {section}")
    list_text = _extract_bracketed_list(stability_text, values_start)
    list_text = re.sub(r"np\.float64\(([-+0-9.eE]+)\)", r"\1", list_text)
    list_text = re.sub(r"(?<![\w.])-inf(?!\w)", "-1e309", list_text)
    list_text = re.sub(r"(?<![\w.])inf(?!\w)", "1e309", list_text)
    list_text = re.sub(r"(?<![\w.])nan(?!\w)", "None", list_text)
    return ast.literal_eval(list_text)


def _as_float(value) -> float:
    return float("nan") if value is None else float(value)


def _task_number(path: Path) -> int:
    match = re.search(r"-task(\d+)_", path.name)
    return int(match.group(1)) if match else 0


def load_unique_element_counts(cache_csv: Path) -> Dict[str, Dict[str, int]]:
    if not cache_csv.exists():
        raise SystemExit(f"Element cache not found: {cache_csv}")
    by_file: Dict[str, int] = {}
    with cache_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            amounts = row.get("element_amounts", "")
            n_unique = 0
            for item in amounts.split(";"):
                if not item:
                    continue
                _, amount = item.split(":", 1)
                if float(amount) > 0:
                    n_unique += 1
            by_file[row["file"]] = n_unique
    return {"by_file": by_file}


def unique_counts_for_model(spec: ModelSpec, cache: Dict[str, Dict[str, int]], limit: int) -> List[Optional[int]]:
    cifs = sorted(spec.cif_dir.glob("*.cif"))[:limit]
    by_file = cache["by_file"]
    return [by_file.get(str(path)) for path in cifs]


def parse_all_metrics(
    specs: Sequence[ModelSpec],
    base_dir: Path,
    element_cache: Path,
    limit: int,
    recursive_json: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache = load_unique_element_counts(element_cache)
    records = []
    summary_records = []
    source_records = []

    for spec in specs:
        json_files, pattern_used = find_json_files(base_dir, spec, recursive=recursive_json)
        json_files = sorted(json_files, key=_task_number)
        n_unique_by_idx = unique_counts_for_model(spec, cache, limit)
        offset = 0
        print(f"{spec.name}: {len(json_files)} JSON files via {pattern_used}")

        for json_path in json_files:
            data = json.load(open(json_path))
            run_n = int(data.get("run_info", {}).get("n_structures", 0))
            valid_ids = data.get("validity_filtering", {}).get("valid_structure_ids", [])
            stability_text = data["results"]["stability"]
            source_records.append(
                {
                    "model": spec.name,
                    "json_file": str(json_path),
                    "pattern_used": pattern_used,
                    "task_number": _task_number(json_path),
                    "n_input_structures": run_n,
                    "n_valid_structures": len(valid_ids),
                    "global_offset": offset,
                }
            )

            for metric_name, cfg in METRICS.items():
                section = cfg["section"]
                mean_key = cfg["mean_key"]
                summary_records.append(
                    {
                        "model": spec.name,
                        "json_file": str(json_path),
                        "metric": metric_name,
                        "mean": _extract_number(stability_text, mean_key),
                        "orb": _extract_number(stability_text, f"{mean_key}_orb"),
                        "mace": _extract_number(stability_text, f"{mean_key}_mace"),
                        "uma": _extract_number(stability_text, f"{mean_key}_uma"),
                        "n_valid_structures": len(valid_ids),
                    }
                )
                values = _extract_individual_values(stability_text, section)
                if len(values) != len(valid_ids):
                    print(
                        f"  warning: {json_path.name} {metric_name} has "
                        f"{len(values)} values but {len(valid_ids)} valid IDs"
                    )
                for pos, item in enumerate(values):
                    local_id = int(valid_ids[pos]) if pos < len(valid_ids) else pos
                    global_idx = offset + local_id
                    n_unique = n_unique_by_idx[global_idx] if global_idx < len(n_unique_by_idx) else None
                    base = {
                        "model": spec.name,
                        "json_file": str(json_path),
                        "task_number": _task_number(json_path),
                        "local_valid_index": local_id,
                        "global_index": global_idx,
                        "unique_elements_per_material": n_unique,
                        "metric": metric_name,
                    }
                    for predictor, key in PREDICTORS:
                        records.append({**base, "predictor": predictor, "value": _as_float(item[key])})
            offset += run_n

    long_df = pd.DataFrame(records)
    summary_df = pd.DataFrame(summary_records)
    source_df = pd.DataFrame(source_records)
    return long_df, summary_df, source_df


def aggregate_model_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, metric, predictor), group in long_df.groupby(["model", "metric", "predictor"], sort=False):
        values = group["value"].dropna()
        rows.append(
            {
                "model": model,
                "metric": metric,
                "predictor": predictor,
                "mean": values.mean(),
                "std": values.std(),
                "median": values.median(),
                "q05": values.quantile(0.05),
                "q95": values.quantile(0.95),
                "n": len(values),
            }
        )
    return pd.DataFrame(rows)


def apply_json_mean_values(model_summary: pd.DataFrame, task_summary: pd.DataFrame) -> pd.DataFrame:
    """Use the benchmark JSON's reported mean for the model-level mean rows.

    The remaining summary columns still describe the filtered per-structure
    values.  Task-level means are combined with valid-structure weighting so
    models represented by multiple JSON files retain the correct overall mean.
    """
    if model_summary.empty or task_summary.empty:
        return model_summary
    required = {"model", "metric", "mean", "n_valid_structures"}
    if not required.issubset(task_summary.columns):
        return model_summary
    task = task_summary.dropna(subset=["mean", "n_valid_structures"]).copy()
    task["_weighted_mean"] = task["mean"] * task["n_valid_structures"]
    json_means = (
        task.groupby(["model", "metric"], as_index=False)
        .agg(_weighted_mean=("_weighted_mean", "sum"), _n_valid=("n_valid_structures", "sum"))
    )
    json_means["_json_mean"] = json_means["_weighted_mean"] / json_means["_n_valid"]
    result = model_summary.copy()
    result = result.merge(json_means[["model", "metric", "_json_mean"]], on=["model", "metric"], how="left")
    mask = (result["predictor"] == "mean") & result["_json_mean"].notna()
    result.loc[mask, "mean"] = result.loc[mask, "_json_mean"]
    return result.drop(columns=["_json_mean"])


def split_metric_outliers(long_df: pd.DataFrame, iqr_multiplier: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if iqr_multiplier <= 0:
        return long_df.copy(), long_df.iloc[0:0].copy()

    clean_parts = []
    outlier_parts = []
    group_cols = ["model", "metric", "predictor"]
    for _, group in long_df.groupby(group_cols, sort=False):
        values = group["value"].dropna()
        if len(values) < 4:
            clean_parts.append(group)
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            clean_parts.append(group)
            continue
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        mask = group["value"].notna() & ((group["value"] < lower) | (group["value"] > upper))
        if mask.any():
            outliers = group[mask].copy()
            outliers["outlier_lower_bound"] = lower
            outliers["outlier_upper_bound"] = upper
            outliers["outlier_iqr_multiplier"] = iqr_multiplier
            outlier_parts.append(outliers)
        clean_parts.append(group[~mask])

    clean_df = pd.concat(clean_parts, ignore_index=True) if clean_parts else long_df.iloc[0:0].copy()
    outlier_df = pd.concat(outlier_parts, ignore_index=True) if outlier_parts else long_df.iloc[0:0].copy()
    return clean_df, outlier_df


def stable_status(mean_ehull: float) -> str:
    if mean_ehull <= 0:
        return "Stable"
    if mean_ehull <= 0.1:
        return "Metastable"
    return "Unstable"


def stable_metastable_distribution(long_df: pd.DataFrame) -> pd.DataFrame:
    ehull = long_df[(long_df["metric"] == "e_above_hull") & (long_df["predictor"] == "mean")].copy()
    ehull["status"] = ehull["value"].map(stable_status)
    ehull = ehull[ehull["status"].isin(["Stable", "Metastable"])]
    ehull["nary_bin"] = ehull["unique_elements_per_material"].map(
        lambda x: "6+" if pd.notna(x) and int(x) >= 6 else (str(int(x)) if pd.notna(x) else "")
    )
    bins = ["1", "2", "3", "4", "5", "6+"]
    records = []
    for (model, status), group in ehull.groupby(["model", "status"], sort=False):
        total = len(group)
        counts = Counter(group["nary_bin"])
        for label in bins:
            count = counts.get(label, 0)
            records.append(
                {
                    "model": model,
                    "status": status,
                    "unique_elements_per_material": label,
                    "count": count,
                    "normalized_percentage": 100.0 * count / total if total else 0.0,
                    "n_structures_in_status": total,
                }
            )
    return pd.DataFrame(records)


def write_outlier_outputs(outlier_df: pd.DataFrame, out_dir: Path) -> None:
    outlier_dir = out_dir / "csv" / "outliers"
    outlier_dir.mkdir(parents=True, exist_ok=True)
    outlier_df.to_csv(outlier_dir / "all_metric_outliers.csv", index=False)
    if outlier_df.empty:
        return
    for model, group in outlier_df.groupby("model", sort=False):
        group.to_csv(outlier_dir / f"{_safe_filename(model)}_metric_outliers.csv", index=False)


def write_outputs(
    raw_long_df: pd.DataFrame,
    filtered_long_df: pd.DataFrame,
    task_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    source_df: pd.DataFrame,
    sm_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    raw_long_df.to_csv(csv_dir / "per_structure_metric_values_long_raw.csv", index=False)
    filtered_long_df.to_csv(csv_dir / "per_structure_metric_values_long.csv", index=False)
    task_summary.to_csv(csv_dir / "task_metric_summary_from_json.csv", index=False)
    model_summary.to_csv(csv_dir / "model_metric_summary_recomputed.csv", index=False)
    source_df.to_csv(csv_dir / "json_sources.csv", index=False)
    sm_df.to_csv(csv_dir / "stable_metastable_vs_unique_elements.csv", index=False)
    write_outlier_outputs(outlier_df, out_dir)


def _axis_style(ax) -> None:
    ax.grid(axis="y", color="#dce7e8", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors="#263238", length=0)
    for spine in ax.spines.values():
        spine.set_color("#263238")
    ax.set_facecolor("#fbfdfd")


def plot_per_model_metric_distributions(long_df: pd.DataFrame, model_order: Sequence[str], out_dir: Path, dpi: int) -> None:
    plot_dir = out_dir / "per_model_metric_distributions"
    plot_dir.mkdir(parents=True, exist_ok=True)
    pcolors = _predictor_colors()
    for model in model_order:
        fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), gridspec_kw={"wspace": 0.24})
        fig.patch.set_facecolor("#fbfdfd")
        fig.suptitle(f"{model}: Multi-MLIP Metric Distributions", fontsize=17, fontweight="bold", y=0.98)
        for ax, (metric, cfg) in zip(axes, METRICS.items()):
            subset = long_df[(long_df["model"] == model) & (long_df["metric"] == metric)]
            for predictor, _ in PREDICTORS:
                if predictor == "mean":
                    continue
                values = subset[subset["predictor"] == predictor]["value"].dropna().to_numpy()
                ax.hist(
                    values,
                    bins=cfg["bins"],
                    density=False,
                    alpha=0.42,
                    color=pcolors[predictor],
                    edgecolor=pcolors[predictor],
                    linewidth=0.7,
                    label=predictor,
                )
            if metric == "e_above_hull":
                ax.axvline(0, color="#b65b5b", linestyle="--", linewidth=1.4, label="Stable")
                ax.axvline(0.1, color="#f2b84b", linestyle="--", linewidth=1.4, label="Metastable")
            ax.set_title(cfg["title"], fontsize=13)
            ax.set_xlabel(cfg["x_label"], fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            _axis_style(ax)
        axes[1].legend(frameon=True, facecolor="white", edgecolor="#d0d8dc", fontsize=9)
        fig.tight_layout()
        output = plot_dir / f"{_safe_filename(model)}_mlip_metric_distributions.png"
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


def plot_mean_metric_comparison(long_df: pd.DataFrame, model_order: Sequence[str], out_dir: Path, dpi: int) -> None:
    plot_dir = out_dir / "mean_metric_comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = _model_colors(model_order)
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.35), gridspec_kw={"wspace": 0.24})
    fig.patch.set_facecolor("#fbfdfd")
    fig.suptitle("Mean Metric Distributions Across Models", fontsize=17, fontweight="bold", y=0.98)
    for ax, (metric, cfg) in zip(axes, METRICS.items()):
        for model in model_order:
            values = long_df[
                (long_df["model"] == model)
                & (long_df["metric"] == metric)
                & (long_df["predictor"] == "mean")
            ]["value"].dropna().to_numpy()
            ax.hist(values, bins=cfg["bins"], alpha=0.32, color=colors[model], edgecolor=colors[model], linewidth=0.65, label=model)
        if metric == "e_above_hull":
            ax.axvline(0, color="#b65b5b", linestyle="--", linewidth=1.5)
            ax.axvline(0.1, color="#f2b84b", linestyle="--", linewidth=1.5)
        ax.set_title(cfg["title"], fontsize=13)
        ax.set_xlabel(cfg["x_label"], fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        _axis_style(ax)
    axes[1].legend(frameon=True, facecolor="white", edgecolor="#d0d8dc", fontsize=8, ncol=1)
    fig.tight_layout()
    output = plot_dir / "combined_mean_metric_distributions.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def load_mp20_formation_energy_mean(mp20_csv: Path) -> Optional[float]:
    if not mp20_csv.exists():
        print(f"Warning: MP20 CSV not found, skipping MP20 formation-energy reference: {mp20_csv}")
        return None
    df = pd.read_csv(mp20_csv, usecols=["formation_energy_per_atom"])
    values = pd.to_numeric(df["formation_energy_per_atom"], errors="coerce").dropna()
    if values.empty:
        print(f"Warning: MP20 CSV has no valid formation_energy_per_atom values: {mp20_csv}")
        return None
    return float(values.mean())


def write_mp20_reference_metrics(mp20_formation_energy_mean: Optional[float], out_dir: Path) -> None:
    if mp20_formation_energy_mean is None:
        return
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model": "MP20",
                "metric": "formation_energy",
                "predictor": "reference",
                "mean": mp20_formation_energy_mean,
                "source_column": "formation_energy_per_atom",
            }
        ]
    ).to_csv(csv_dir / "mp20_reference_metrics.csv", index=False)


def plot_model_metric_summary_bars(
    model_summary: pd.DataFrame,
    model_order: Sequence[str],
    out_dir: Path,
    dpi: int,
    mp20_formation_energy_mean: Optional[float] = None,
) -> None:
    plot_dir = out_dir / "mean_metric_comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)
    mean_df = model_summary[model_summary["predictor"] == "mean"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.4), gridspec_kw={"wspace": 0.25})
    fig.patch.set_facecolor("#fbfdfd")
    colors = _model_colors([*model_order, "MP20"])
    colors["MP20"] = "#37474f"
    for ax, (metric, cfg) in zip(axes, METRICS.items()):
        labels = list(model_order)
        vals = []
        for model in model_order:
            row = mean_df[(mean_df["model"] == model) & (mean_df["metric"] == metric)]
            vals.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
        if metric == "formation_energy" and mp20_formation_energy_mean is not None:
            labels.append("MP20")
            vals.append(mp20_formation_energy_mean)
        x = np.arange(len(labels))
        ax.bar(x, vals, color=[colors[m] for m in labels], edgecolor="white", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
        ax.set_title(f"Mean {cfg['title']}", fontsize=13)
        ax.set_ylabel(cfg["x_label"], fontsize=10)
        _axis_style(ax)
    fig.suptitle("Model-Level Mean Metrics", fontsize=17, fontweight="bold", y=0.98)
    fig.tight_layout()
    output = plot_dir / "model_level_mean_metrics_bar.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_stable_metastable(
    sm_df: pd.DataFrame,
    model_order: Sequence[str],
    out_dir: Path,
    dpi: int,
    per_model_order: Optional[Sequence[str]] = None,
) -> None:
    plot_dir = out_dir / "stable_metastable_unique_elements"
    plot_dir.mkdir(parents=True, exist_ok=True)
    bins = ["1", "2", "3", "4", "5", "6+"]
    status_colors = {"Stable": "#b65b5b", "Metastable": "#f2b84b"}
    for model in per_model_order or model_order:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        fig.patch.set_facecolor("#fbfdfd")
        x = np.arange(len(bins))
        width = 0.34
        for offset, status in [(-width / 2, "Stable"), (width / 2, "Metastable")]:
            subset = sm_df[(sm_df["model"] == model) & (sm_df["status"] == status)].set_index("unique_elements_per_material")
            values = [float(subset.loc[b, "normalized_percentage"]) if b in subset.index else 0.0 for b in bins]
            ax.bar(x + offset, values, width=width, label=status, color=status_colors[status], edgecolor="white", linewidth=0.7, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(bins, fontsize=12)
        ax.set_xlabel("Unique Elements per Material", fontsize=13)
        ax.set_ylabel("Normalized Percentage", fontsize=13)
        ax.set_title(f"{model}: Stable/Metastable N-ary Distribution", fontsize=14, fontweight="bold")
        ax.legend(frameon=False, fontsize=10)
        _axis_style(ax)
        fig.tight_layout()
        output = plot_dir / f"{_safe_filename(model)}_stable_metastable_unique_elements.png"
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.0), gridspec_kw={"wspace": 0.22})
    fig.patch.set_facecolor("#fbfdfd")
    model_colors = _model_colors(model_order)
    for ax, status in zip(axes, ["Stable", "Metastable"]):
        x = np.arange(len(bins), dtype=float)
        width = min(0.11, 0.82 / max(1, len(model_order)))
        offsets = (np.arange(len(model_order)) - (len(model_order) - 1) / 2.0) * width
        for offset, model in zip(offsets, model_order):
            subset = sm_df[(sm_df["model"] == model) & (sm_df["status"] == status)].set_index("unique_elements_per_material")
            values = [float(subset.loc[b, "normalized_percentage"]) if b in subset.index else 0.0 for b in bins]
            ax.bar(x + offset, values, width=width, color=model_colors[model], edgecolor="white", linewidth=0.6, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(bins, fontsize=11)
        ax.set_xlabel("Unique Elements per Material", fontsize=12)
        ax.set_ylabel("Normalized Percentage", fontsize=12)
        ax.set_title(status, fontsize=14, fontweight="bold")
        _axis_style(ax)
    axes[1].legend(frameon=False, fontsize=8, ncol=1)
    fig.suptitle("Stable and Metastable N-ary Distributions", fontsize=17, fontweight="bold", y=0.98)
    fig.tight_layout()
    output = plot_dir / "combined_stable_metastable_unique_elements.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze FieldForge paper comparison MLIP metrics.")
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "mlip_metrics")
    parser.add_argument("--element-cache", type=Path, default=ELEMENT_CACHE)
    parser.add_argument(
        "--mp20-csv",
        type=Path,
        default=MP20_CSV,
        help="MP20 merged CSV used to add formation_energy_per_atom mean to model_level_mean_metrics_bar.png.",
    )
    parser.add_argument(
        "--no-mp20-formation-reference",
        action="store_true",
        help="Do not add the MP20 formation_energy_per_atom mean to the formation-energy summary bar plot.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(MODEL_PRESETS),
        default="main",
        help=(
            "Built-in JSON/CIF model list to analyze. 'main' defaults to "
            "FieldForge (PT) and Crystalite."
        ),
    )
    parser.add_argument(
        "--include-model",
        action="append",
        default=[],
        help=(
            "Append a registered model to the selected preset. Repeatable. "
            f"Registered models: {', '.join(MODEL_REGISTRY)}"
        ),
    )
    parser.add_argument(
        "--only-model",
        action="append",
        default=[],
        help=(
            "Use only this registered model, in the order provided. Repeatable. "
            "When set, --preset is used only for defaults not covered by custom --model entries."
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("NAME", "JSON_GLOB", "CIF_DIR"),
        help="Append a custom model by name, JSON glob pattern, and CIF directory.",
    )
    parser.add_argument(
        "--no-recursive-json",
        action="store_true",
        help="Only search --base-dir itself for JSON files; by default subdirectories are searched too.",
    )
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument(
        "--outlier-iqr-multiplier",
        type=float,
        default=3.0,
        help="Drop metric values outside Q1/Q3 +/- multiplier*IQR before plotting; <=0 disables filtering.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _configure_style()
    specs = load_model_specs(
        args.preset,
        only_models=args.only_model,
        include_models=args.include_model,
        custom_models=args.model,
    )
    model_order = [spec.name for spec in specs]
    raw_long_df, task_summary, source_df = parse_all_metrics(
        specs, args.base_dir, args.element_cache, args.limit, recursive_json=not args.no_recursive_json
    )
    filtered_long_df, outlier_df = split_metric_outliers(raw_long_df, args.outlier_iqr_multiplier)
    model_summary = apply_json_mean_values(aggregate_model_summary(filtered_long_df), task_summary)
    sm_df = stable_metastable_distribution(filtered_long_df)
    mp20_formation_energy_mean = None
    if not args.no_mp20_formation_reference:
        mp20_formation_energy_mean = load_mp20_formation_energy_mean(args.mp20_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(raw_long_df, filtered_long_df, task_summary, model_summary, source_df, sm_df, outlier_df, args.output_dir)
    write_mp20_reference_metrics(mp20_formation_energy_mean, args.output_dir)
    plot_per_model_metric_distributions(filtered_long_df, model_order, args.output_dir, args.dpi)
    plot_mean_metric_comparison(filtered_long_df, model_order, args.output_dir, args.dpi)
    plot_model_metric_summary_bars(model_summary, model_order, args.output_dir, args.dpi, mp20_formation_energy_mean)
    plot_stable_metastable(sm_df, model_order, args.output_dir, args.dpi)
    print(f"Dropped {len(outlier_df)} outlier metric values with IQR multiplier {args.outlier_iqr_multiplier:g}.")
    print(f"Wrote analysis outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
