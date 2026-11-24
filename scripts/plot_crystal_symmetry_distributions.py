#!/usr/bin/env python3
"""Plot crystallographic symmetry-property distributions for generated CIFs.

The script analyzes CIF files with pymatgen's SpacegroupAnalyzer and produces a
publication-style figure inspired by stacked distribution bars in crystal
generation papers.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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

from settings import CIF_MODELS, OUTPUT_ROOT

MAIN_MODELS = CIF_MODELS
MODEL_PRESETS = {"main": MAIN_MODELS}
DEFAULT_MODELS = MAIN_MODELS


@dataclass(frozen=True)
class ModelInput:
    name: str
    path: Path


def _require_pymatgen():
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: pymatgen. Create an isolated environment, for example:\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  python -m pip install -U pip\n"
            "  python -m pip install -r requirements.txt\n"
            "Then rerun this script with `.venv/bin/python`."
        ) from exc
    return Structure, SpacegroupAnalyzer


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


def _frac_close_to_integer(values: np.ndarray, tol: float) -> bool:
    return bool(np.all(np.abs(values - np.round(values)) <= tol))


def _wyckoff_dimension(frac_coord: Sequence[float], symm_ops, tol: float = 1e-5) -> int:
    """Return positional degrees of freedom for a Wyckoff orbit representative.

    Operations that map the representative coordinate back onto itself modulo a
    lattice vector form the stabilizer. Infinitesimal displacements must lie in
    the null space of (R - I) for every stabilizer rotation R; the nullity is the
    Wyckoff dimensionality in {0, 1, 2, 3}.
    """
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

    matrix = np.vstack(constraints)
    rank = int(np.linalg.matrix_rank(matrix, tol=1e-7))
    return max(0, min(3, 3 - rank))


def _dataset_value(dataset, key: str):
    if hasattr(dataset, key):
        return getattr(dataset, key)
    return dataset[key]


def _analyze_one(args: Tuple[str, str, float, float]) -> List[Dict[str, str]]:
    model_name, cif_path, symprec, angle_tolerance = args
    Structure, SpacegroupAnalyzer = _require_pymatgen()

    path = Path(cif_path)
    rows: List[Dict[str, str]] = []
    try:
        structure = Structure.from_file(str(path), primitive=False, sort=False)
        sga = SpacegroupAnalyzer(
            structure, symprec=symprec, angle_tolerance=angle_tolerance
        )
        dataset = sga.get_symmetry_dataset()
        symm_ops = sga.get_space_group_operations()

        sg_number = int(_dataset_value(dataset, "number"))
        international = str(_dataset_value(dataset, "international"))
        hall = str(_dataset_value(dataset, "hall"))
        wyckoffs = list(_dataset_value(dataset, "wyckoffs"))
        equivalent_atoms = np.asarray(_dataset_value(dataset, "equivalent_atoms"))

        rows.append(
            {
                "record_type": "structure",
                "model": model_name,
                "file": str(path),
                "spacegroup_number": str(sg_number),
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
            if len(indices) == 0:
                continue
            rep_index = int(indices[0])
            frac = structure[rep_index].frac_coords
            dim = _wyckoff_dimension(frac, symm_ops)
            rows.append(
                {
                    "record_type": "wyckoff",
                    "model": model_name,
                    "file": str(path),
                    "spacegroup_number": str(sg_number),
                    "spacegroup_symbol": international,
                    "hall_symbol": hall,
                    "wyckoff_letter": str(wyckoffs[rep_index]),
                    "multiplicity": str(len(indices)),
                    "dimensionality": str(dim),
                    "status": "ok",
                    "error": "",
                }
            )
    except Exception as exc:  # noqa: BLE001 - keep batch jobs moving.
        rows.append(
            {
                "record_type": "error",
                "model": model_name,
                "file": str(path),
                "spacegroup_number": "",
                "spacegroup_symbol": "",
                "hall_symbol": "",
                "wyckoff_letter": "",
                "multiplicity": "",
                "dimensionality": "",
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return rows


def analyze_models(
    models: Sequence[ModelInput],
    symprec: float,
    angle_tolerance: float,
    recursive: bool,
    limit: Optional[int],
    workers: int,
    backend: str,
) -> List[Dict[str, str]]:
    jobs = []
    for model in models:
        cifs = _find_cifs(model.path, recursive=recursive, limit=limit)
        if not cifs:
            raise SystemExit(f"No CIF files found for {model.name}: {model.path}")
        jobs.extend((model.name, str(cif), symprec, angle_tolerance) for cif in cifs)

    all_rows: List[Dict[str, str]] = []
    total = len(jobs)
    print(
        f"Analyzing {total} CIF files with {workers} worker(s), backend={backend}...",
        flush=True,
    )

    if workers == 1:
        for idx, job in enumerate(jobs, 1):
            all_rows.extend(_analyze_one(job))
            if idx % 100 == 0 or idx == total:
                print(f"  {idx}/{total} files processed", flush=True)
    else:
        executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
        with executor_cls(max_workers=workers) as executor:
            futures = [executor.submit(_analyze_one, job) for job in jobs]
            for idx, future in enumerate(as_completed(futures), 1):
                all_rows.extend(future.result())
                if idx % 100 == 0 or idx == total:
                    print(f"  {idx}/{total} files processed", flush=True)

    return all_rows


def write_csv(rows: Sequence[Dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_filename(name: str) -> str:
    safe = []
    for char in name:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "model"


def write_per_model_csvs(
    rows: Sequence[Dict[str, str]], model_order: Sequence[str], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for model in model_order:
        model_rows = [row for row in rows if row["model"] == model]
        output_csv = output_dir / f"{_safe_filename(model)}_symmetry_records.csv"
        write_csv(model_rows, output_csv)
        print(f"Wrote per-model CSV: {output_csv}")


def read_csv(input_csv: Path) -> List[Dict[str, str]]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _spacegroup_symbol_from_label(label: str) -> str:
    parts = label.split(" ", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]
    return label


def _spacegroup_number_from_label(label: str) -> str:
    parts = label.split(" ", 1)
    return parts[0] if parts and parts[0].isdigit() else ""


def _sort_spacegroups(labels: Iterable[str]) -> List[str]:
    def key(label: str) -> Tuple[int, str]:
        try:
            return (int(label.split(" ", 1)[0]), label)
        except ValueError:
            return (9999, label)

    return sorted(labels, key=key)


def _count_property(
    rows: Sequence[Dict[str, str]], model_order: Sequence[str], prop: str
) -> Dict[str, Counter]:
    counts: Dict[str, Counter] = {model: Counter() for model in model_order}
    for row in rows:
        model = row["model"]
        if model not in counts:
            continue
        if prop == "spacegroup":
            if row["record_type"] != "structure":
                continue
            label = f"{row['spacegroup_number']} {row['spacegroup_symbol']}".strip()
        else:
            if row["record_type"] != "wyckoff":
                continue
            label = row[prop]
        if label:
            counts[model][label] += 1
    return counts


def write_spacegroup_mapping_excel(
    rows: Sequence[Dict[str, str]],
    model_order: Sequence[str],
    output: Path,
) -> None:
    counts = _count_property(rows, model_order, "spacegroup")
    labels = _sort_spacegroups({label for counter in counts.values() for label in counter})
    records = []
    for label in labels:
        record = {
            "spacegroup_number": _spacegroup_number_from_label(label),
            "spacegroup_symbol": _spacegroup_symbol_from_label(label),
            "plot_label": _spacegroup_symbol_from_label(label),
            "full_label": label,
        }
        total = 0
        for model in model_order:
            value = counts[model].get(label, 0)
            model_total = sum(counts[model].values())
            total += value
            record[f"{model}_count"] = value
            record[f"{model}_frequency_percent"] = 100.0 * value / model_total if model_total else 0.0
        record["total_count"] = total
        records.append(record)

    output.parent.mkdir(parents=True, exist_ok=True)
    mapping = pd.DataFrame(records)
    try:
        mapping.to_excel(output, index=False)
        print(f"Wrote Spacegroup mapping Excel: {output}")
    except ImportError:
        fallback = output.with_suffix(".csv")
        mapping.to_csv(fallback, index=False)
        print(f"Missing Excel writer; wrote Spacegroup mapping CSV: {fallback}")


def _compress_labels(
    counts: Dict[str, Counter], max_categories: Optional[int], sort_kind: str
) -> Tuple[Dict[str, Counter], List[str]]:
    all_counts = Counter()
    for counter in counts.values():
        all_counts.update(counter)

    if sort_kind == "spacegroup":
        ordered = _sort_spacegroups(all_counts)
    elif sort_kind == "numeric":
        ordered = sorted(all_counts, key=lambda x: (float(x), x))
    else:
        ordered = [label for label, _ in all_counts.most_common()]

    if max_categories is not None and len(ordered) > max_categories:
        keep = set([label for label, _ in all_counts.most_common(max_categories)])
        compressed: Dict[str, Counter] = {}
        for model, counter in counts.items():
            new_counter = Counter()
            other = 0
            for label, value in counter.items():
                if label in keep:
                    new_counter[label] += value
                else:
                    other += value
            if other:
                new_counter["Other"] = other
            compressed[model] = new_counter
        ordered = _sort_spacegroups(keep) if sort_kind == "spacegroup" else sorted(keep)
        ordered.append("Other")
        return compressed, ordered

    return counts, ordered


def _make_palette(n: int) -> List[Tuple[float, float, float, float]]:
    base_maps = ["tab20", "Set3", "Paired", "Accent", "Dark2"]
    colors = []
    for cmap_name in base_maps:
        cmap = mpl.colormaps[cmap_name]
        size = getattr(cmap, "N", 12)
        colors.extend(cmap(i / max(1, size - 1)) for i in range(size))
    if n <= len(colors):
        return colors[:n]
    extra = mpl.colormaps["turbo"](np.linspace(0.05, 0.95, n - len(colors)))
    colors.extend(list(extra))
    return colors[:n]


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


def _plot_stacked_bars(
    ax,
    counts: Dict[str, Counter],
    labels: Sequence[str],
    model_order: Sequence[str],
    title: str,
    colors: Sequence[Tuple[float, float, float, float]],
    show_ylabels: bool,
) -> None:
    y_positions = np.arange(len(model_order))
    for y, model in zip(y_positions, model_order):
        total = sum(counts[model].values())
        left = 0.0
        for label, color in zip(labels, colors):
            value = counts[model].get(label, 0)
            if total <= 0 or value <= 0:
                continue
            width = value / total
            ax.barh(
                y,
                width,
                left=left,
                height=0.86,
                color=color,
                edgecolor="white",
                linewidth=0.7,
            )
            left += width

    ax.set_xlim(0, 1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_order if show_ylabels else [], fontsize=11)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=15, pad=4, color="#253238")
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", length=0, colors="#37474f")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#fbfdfd")


def plot_figure(
    rows: Sequence[Dict[str, str]],
    model_order: Sequence[str],
    output: Path,
    max_spacegroups: int,
    title: str,
    dpi: int,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    properties = [
        ("spacegroup", "Spacegroup", max_spacegroups, "spacegroup"),
        ("multiplicity", "Wyckoff multiplicity", None, "numeric"),
        ("dimensionality", "Wyckoff dimensionality", None, "numeric"),
    ]

    many_models = len(model_order) > 4
    top_margin = 0.78 if many_models else 0.72
    title_y = 0.985 if many_models else 0.93
    rule_y = 0.84
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.8, max(3.55, 0.62 * len(model_order) + 2.25)),
        gridspec_kw={
            "wspace": 0.045,
            "left": 0.08,
            "right": 0.985,
            "top": top_margin,
            "bottom": 0.18,
        },
    )
    fig.patch.set_facecolor("#fbfdfd")
    if not many_models:
        fig.suptitle(title, fontsize=18, fontweight="bold", y=title_y)
    if not many_models:
        fig.add_artist(
            mpl.lines.Line2D(
                [0.025, 0.985],
                [rule_y, rule_y],
                transform=fig.transFigure,
                color="#1c2428",
                lw=1.6,
            )
        )

    for ax_idx, (ax, (prop, prop_title, max_categories, sort_kind)) in enumerate(
        zip(axes, properties)
    ):
        counts = _count_property(rows, model_order, prop)
        compressed, labels = _compress_labels(counts, max_categories, sort_kind)
        colors = _make_palette(len(labels))
        if labels and labels[-1] == "Other":
            colors[-1] = (0.78, 0.82, 0.86, 1.0)
        _plot_stacked_bars(
            ax,
            compressed,
            labels,
            model_order,
            prop_title,
            colors,
            show_ylabels=ax_idx == 0,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _top_labels_by_frequency(counts: Dict[str, Counter], limit: Optional[int]) -> List[str]:
    all_counts = Counter()
    for counter in counts.values():
        all_counts.update(counter)
    labels = [label for label, _ in all_counts.most_common(limit)]
    return labels


def _top_labels_by_reference(
    counts: Dict[str, Counter], limit: Optional[int], reference_model: Optional[str] = None
) -> List[str]:
    if reference_model is None:
        return _top_labels_by_frequency(counts, limit)
    if reference_model not in counts:
        raise ValueError(f"Reference model not found in counts: {reference_model}")
    labels = [label for label, _ in counts[reference_model].most_common(limit)]
    if not labels:
        return _top_labels_by_frequency(counts, limit)
    return labels


def _normalised_values(
    counts: Dict[str, Counter], model: str, labels: Sequence[str]
) -> List[float]:
    total = sum(counts[model].values())
    if total == 0:
        return [0.0 for _ in labels]
    return [100.0 * counts[model].get(label, 0) / total for label in labels]


def _plot_grouped_panel(
    ax,
    counts: Dict[str, Counter],
    labels: Sequence[str],
    model_order: Sequence[str],
    title: str,
    model_colors: Dict[str, Tuple[float, float, float, float]],
    rotate_labels: bool,
) -> None:
    x = np.arange(len(labels), dtype=float)
    width = min(0.24, 0.82 / max(1, len(model_order)))
    offsets = (np.arange(len(model_order)) - (len(model_order) - 1) / 2.0) * width

    for offset, model in zip(offsets, model_order):
        values = _normalised_values(counts, model, labels)
        ax.bar(
            x + offset,
            values,
            width=width,
            label=model,
            color=model_colors[model],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
        )

    ax.set_title(title, fontsize=15, pad=10, color="#253238")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_spacegroup_symbol_from_label(label) for label in labels],
        fontsize=8.5 if rotate_labels else 10,
        rotation=38 if rotate_labels else 0,
        ha="right" if rotate_labels else "center",
    )
    ax.set_ylabel("Frequency (%)", fontsize=11, color="#37474f")
    ax.grid(axis="y", color="#dce7e8", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors="#37474f", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd7da")
    ax.spines["bottom"].set_color("#cbd7da")
    ax.set_facecolor("#fbfdfd")


def plot_grouped_bar_figure(
    rows: Sequence[Dict[str, str]],
    model_order: Sequence[str],
    output: Path,
    top_spacegroups: int,
    title: str,
    dpi: int,
    reference_model: Optional[str] = None,
) -> None:
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
    panels = [
        ("spacegroup", "Top Spacegroups", top_spacegroups, True),
        ("multiplicity", "Wyckoff multiplicity", None, False),
        ("dimensionality", "Wyckoff dimensionality", None, False),
    ]

    top_fraction = 0.58 if len(model_order) > 4 else 0.68
    fig_height = 6.0 if len(model_order) > 4 else 5.25
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.4, fig_height),
        gridspec_kw={
            "wspace": 0.24,
            "left": 0.06,
            "right": 0.99,
            "top": top_fraction,
            "bottom": 0.22,
        },
    )
    fig.patch.set_facecolor("#fbfdfd")
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.95)
    fig.add_artist(
        mpl.lines.Line2D(
            [0.025, 0.985],
            [0.86, 0.86],
            transform=fig.transFigure,
            color="#1c2428",
            lw=1.5,
        )
    )

    for ax, (prop, panel_title, limit, rotate_labels) in zip(axes, panels):
        counts = _count_property(rows, model_order, prop)
        if prop == "spacegroup":
            labels = _top_labels_by_reference(counts, limit, reference_model)
        elif prop in {"multiplicity", "dimensionality"}:
            labels = sorted(
                _top_labels_by_frequency(counts, limit),
                key=lambda value: (float(value), value),
            )
        else:
            labels = _top_labels_by_frequency(counts, limit)
        _plot_grouped_panel(
            ax,
            counts,
            labels,
            model_order,
            panel_title,
            model_colors,
            rotate_labels=rotate_labels,
        )

    handles = [
        mpl.patches.Patch(facecolor=model_colors[model], edgecolor="white", label=model)
        for model in model_order
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.81),
        ncol=min(4, len(model_order)),
        frameon=False,
        fontsize=11,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _style_horizontal_axis(ax, xlabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=11, color="#37474f")
    ax.grid(axis="x", color="#dce7e8", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors="#37474f", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd7da")
    ax.spines["bottom"].set_color("#cbd7da")
    ax.set_facecolor("#fbfdfd")


def plot_spacegroup_comparison_horizontal(
    rows: Sequence[Dict[str, str]],
    model_order: Sequence[str],
    output: Path,
    top_spacegroups: int,
    title: str,
    dpi: int,
    reference_model: Optional[str] = None,
    legend_mode: str = "top",
    close_top_right: bool = False,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    counts = _count_property(rows, model_order, "spacegroup")
    labels = _top_labels_by_reference(counts, top_spacegroups, reference_model)
    model_colors = _model_colors(model_order)
    y = np.arange(len(labels), dtype=float)
    height = min(0.11, 0.82 / max(1, len(model_order)))
    offsets = (np.arange(len(model_order)) - (len(model_order) - 1) / 2.0) * height

    fig_height = max(5.2, 0.34 * len(labels) + 2.4)
    fig, ax = plt.subplots(figsize=(10.4, fig_height))
    fig.patch.set_facecolor("#fbfdfd")
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.985)

    for offset, model in zip(offsets, model_order):
        values = _normalised_values(counts, model, labels)
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
    ax.set_yticklabels([_spacegroup_symbol_from_label(label) for label in labels], fontsize=10)
    ax.invert_yaxis()
    _style_horizontal_axis(ax, "Frequency (%)")
    if close_top_right:
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_color("#cbd7da")
        ax.spines["right"].set_color("#cbd7da")
        ax.tick_params(top=False, right=False, labeltop=False, labelright=False)
        xmax = ax.get_xlim()[1]
        ax.set_xlim(0, xmax * 1.22)
    handles = [
        mpl.patches.Patch(facecolor=model_colors[model], edgecolor="white", label=model)
        for model in model_order
    ]
    if legend_mode == "right":
        ax.legend(
            handles=handles,
            loc="center right",
            bbox_to_anchor=(0.985, 0.54),
            ncol=1,
            frameon=True,
            facecolor="white",
            edgecolor="#cbd7da",
            fontsize=9.5,
        )
        fig.subplots_adjust(left=0.15, right=0.985, bottom=0.09, top=0.91)
    else:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=min(4, len(model_order)),
            frameon=False,
            fontsize=10,
        )
        fig.subplots_adjust(left=0.15, right=0.985, bottom=0.09, top=0.79)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_individual_top_spacegroups(
    rows: Sequence[Dict[str, str]],
    model_order: Sequence[str],
    output_dir: Path,
    top_spacegroups: int,
    dpi: int,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = _count_property(rows, model_order, "spacegroup")
    model_colors = _model_colors(model_order)

    for model in model_order:
        labels = [label for label, _ in counts[model].most_common(top_spacegroups)]
        values = _normalised_values(counts, model, labels)
        y = np.arange(len(labels), dtype=float)
        fig_height = max(4.8, 0.34 * len(labels) + 1.7)
        fig, ax = plt.subplots(figsize=(7.0, fig_height))
        fig.patch.set_facecolor("#fbfdfd")
        ax.barh(
            y,
            values,
            height=0.72,
            color=model_colors[model],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.96,
        )
        ax.set_yticks(y)
        ax.set_yticklabels([_spacegroup_symbol_from_label(label) for label in labels], fontsize=10)
        ax.invert_yaxis()
        ax.set_title(f"{model}: Top {top_spacegroups} Spacegroups", fontsize=15, fontweight="bold", pad=12)
        _style_horizontal_axis(ax, "Frequency (%)")

        xmax = max(values) if values else 0.0
        ax.set_xlim(0, xmax * 1.13 if xmax > 0 else 1)
        for yi, value in zip(y, values):
            ax.text(
                value + max(xmax, 1.0) * 0.015,
                yi,
                f"{value:.1f}%",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#37474f",
            )

        fig.tight_layout()
        output = output_dir / f"{_safe_filename(model)}_top{top_spacegroups}_spacegroups.png"
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Wrote individual Spacegroup chart: {output}")


def summarize(rows: Sequence[Dict[str, str]], model_order: Sequence[str]) -> None:
    errors = [row for row in rows if row["record_type"] == "error"]
    print("\nSummary")
    for model in model_order:
        structures = sum(
            1
            for row in rows
            if row["model"] == model and row["record_type"] == "structure"
        )
        wyckoffs = sum(
            1 for row in rows if row["model"] == model and row["record_type"] == "wyckoff"
        )
        model_errors = sum(
            1 for row in errors if row["model"] == model and row["record_type"] == "error"
        )
        print(
            f"  {model}: {structures} structures, {wyckoffs} Wyckoff orbits, "
            f"{model_errors} errors"
        )
    if errors:
        print(f"  First error: {errors[0]['file']} -> {errors[0]['error']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze CIF symmetry properties and plot stacked distributions."
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("NAME", "CIF_DIR"),
        help=(
            "Model name and CIF directory. Repeat this option for multiple models. "
            "If omitted, the selected --preset model list is used."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=sorted(MODEL_PRESETS),
        default="main",
        help="Built-in model list used when --model is omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "symmetry/figures/symmetry_distributions.png",
        help="Output image path. A matching PDF is also written for non-PDF outputs.",
    )
    parser.add_argument(
        "--bar-output",
        type=Path,
        default=OUTPUT_ROOT / "symmetry/figures/grouped_bars_top15.png",
        help="Output path for grouped bar-chart comparison. A matching PDF is also written.",
    )
    parser.add_argument(
        "--spacegroup-comparison-output",
        type=Path,
        default=None,
        help="Optional horizontal grouped bar chart for top Spacegroups with labels on the y-axis.",
    )
    parser.add_argument(
        "--cache-csv",
        type=Path,
        default=OUTPUT_ROOT / "symmetry/csv/symmetry_records.csv",
        help="CSV cache for per-structure and per-Wyckoff records.",
    )
    parser.add_argument(
        "--per-model-csv-dir",
        type=Path,
        default=None,
        help="Optional directory for one CSV file per model.",
    )
    parser.add_argument(
        "--spacegroup-mapping-output",
        type=Path,
        default=OUTPUT_ROOT / "symmetry/csv/spacegroup_mapping.xlsx",
        help="Excel file recording full Spacegroup number/symbol labels used by plots.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Skip CIF analysis and plot from --cache-csv.",
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=0.1,
        help="Distance tolerance passed to pymatgen SpacegroupAnalyzer.",
    )
    parser.add_argument(
        "--angle-tolerance",
        type=float,
        default=5.0,
        help="Angle tolerance passed to pymatgen SpacegroupAnalyzer.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--backend",
        choices=["thread", "process"],
        default="thread",
        help=(
            "Parallel backend. Use thread inside sandboxed desktop environments; "
            "process can be faster in unrestricted shells."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search CIF files recursively under each model directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of CIFs per model, useful for quick tests.",
    )
    parser.add_argument(
        "--max-spacegroups",
        type=int,
        default=32,
        help="Show this many most frequent space groups; merge the rest into Other.",
    )
    parser.add_argument(
        "--bar-top-spacegroups",
        type=int,
        default=15,
        help="Number of most frequent space groups shown in the grouped bar chart.",
    )
    parser.add_argument(
        "--individual-spacegroup-dir",
        type=Path,
        default=None,
        help="Optional directory for one horizontal top-Spacegroup chart per model.",
    )
    parser.add_argument(
        "--title",
        default="Crystallographic Symmetry Distributions of Generated Structures",
        help="Figure title.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Raster output DPI.")
    parser.add_argument(
        "--skip-stacked",
        action="store_true",
        help="Do not write the stacked distribution figure.",
    )
    parser.add_argument(
        "--skip-bars",
        action="store_true",
        help="Do not write the grouped bar-chart figure.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    models = _parse_model_specs(args.model, args.preset)
    model_order = [model.name for model in models]

    if args.use_cache:
        if not args.cache_csv.exists():
            raise SystemExit(f"Cache file not found: {args.cache_csv}")
        rows = read_csv(args.cache_csv)
    else:
        _require_pymatgen()
        rows = analyze_models(
            models=models,
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance,
            recursive=args.recursive,
            limit=args.limit,
            workers=max(1, args.workers),
            backend=args.backend,
        )
        write_csv(rows, args.cache_csv)
        print(f"Wrote analysis cache: {args.cache_csv}")

    summarize(rows, model_order)
    if args.spacegroup_mapping_output is not None:
        write_spacegroup_mapping_excel(rows, model_order, args.spacegroup_mapping_output)
    if args.per_model_csv_dir is not None:
        write_per_model_csvs(rows, model_order, args.per_model_csv_dir)
    if not args.skip_stacked:
        plot_figure(
            rows=rows,
            model_order=model_order,
            output=args.output,
            max_spacegroups=args.max_spacegroups,
            title=args.title,
            dpi=args.dpi,
        )
        print(f"Wrote stacked figure: {args.output}")
        if args.output.suffix.lower() != ".pdf":
            print(f"Wrote stacked vector copy: {args.output.with_suffix('.pdf')}")
    if not args.skip_bars:
        plot_grouped_bar_figure(
            rows=rows,
            model_order=model_order,
            output=args.bar_output,
            top_spacegroups=args.bar_top_spacegroups,
            title="Grouped Comparison of Crystallographic Symmetry Properties",
            dpi=args.dpi,
        )
        print(f"Wrote grouped bar figure: {args.bar_output}")
        if args.bar_output.suffix.lower() != ".pdf":
            print(f"Wrote grouped bar vector copy: {args.bar_output.with_suffix('.pdf')}")
    if args.spacegroup_comparison_output is not None:
        plot_spacegroup_comparison_horizontal(
            rows=rows,
            model_order=model_order,
            output=args.spacegroup_comparison_output,
            top_spacegroups=args.bar_top_spacegroups,
            title=f"Top {args.bar_top_spacegroups} Spacegroup Comparison",
            dpi=args.dpi,
        )
        print(f"Wrote horizontal Spacegroup comparison: {args.spacegroup_comparison_output}")
        if args.spacegroup_comparison_output.suffix.lower() != ".pdf":
            print(
                "Wrote horizontal Spacegroup vector copy: "
                f"{args.spacegroup_comparison_output.with_suffix('.pdf')}"
            )
    if args.individual_spacegroup_dir is not None:
        plot_individual_top_spacegroups(
            rows=rows,
            model_order=model_order,
            output_dir=args.individual_spacegroup_dir,
            top_spacegroups=args.bar_top_spacegroups,
            dpi=args.dpi,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
