#!/usr/bin/env python3
"""Select and visualize representative novel/unique stable/metastable CIFs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("CRYSTAL_BENCHMARK_OUTPUT_DIR", "outputs")) / ".mplconfig"),
)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Element, Structure

import analyze_mlip_relaxation_metrics as mlip
import plot_crystal_symmetry_distributions as sgplot
from settings import FIELD_FORGE_NAME, OUTPUT_ROOT


BASE_DIR = mlip.BASE_DIR
DEFAULT_SYMMETRY_CSV = (
    OUTPUT_ROOT / "mp20/symmetry_with_mp20/combined_models_with_mp20_symmetry_records.csv"
)
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "representative_structures"
DEFAULT_MODELS = [FIELD_FORGE_NAME]


@dataclass(frozen=True)
class Candidate:
    model: str
    global_index: int
    local_index: int
    json_file: str
    cif_path: Path
    spacegroup_label: str
    spacegroup_symbol: str
    status: str
    novelty: float
    unique: bool
    mean_e_above_hull: float


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "item"


def _extract_list(text: str, marker: str = "individual_values="):
    start = text.find(marker)
    if start < 0:
        return []
    list_text = mlip._extract_bracketed_list(text, start)
    list_text = re.sub(r"np\.float64\(([-+0-9.eE]+)\)", r"\1", list_text)
    list_text = re.sub(r"(?<![\w.])-inf(?!\w)", "-1e309", list_text)
    list_text = re.sub(r"(?<![\w.])inf(?!\w)", "1e309", list_text)
    list_text = re.sub(r"(?<![\w.])nan(?!\w)", "None", list_text)
    return ast.literal_eval(list_text)


def _spacegroup_symbol(label: str) -> str:
    return sgplot._spacegroup_symbol_from_label(label)


def load_spacegroup_by_file(symmetry_csv: Path) -> Dict[str, str]:
    mapping = {}
    with symmetry_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("record_type") != "structure" or row.get("status") != "ok":
                continue
            mapping[row["file"]] = f"{row['spacegroup_number']} {row['spacegroup_symbol']}".strip()
    return mapping


def top_spacegroups_from_reference(symmetry_csv: Path, reference_model: str, n: int) -> List[str]:
    counter = Counter()
    with symmetry_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") != reference_model or row.get("record_type") != "structure":
                continue
            if row.get("status") != "ok":
                continue
            counter[f"{row['spacegroup_number']} {row['spacegroup_symbol']}".strip()] += 1
    return [label for label, _ in counter.most_common(n)]


def normalize_spacegroup_inputs(values: Optional[Sequence[str]], default_labels: Sequence[str]) -> List[str]:
    if not values:
        return list(default_labels)
    wanted = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        wanted.append(value)
    return wanted


def matches_spacegroup(label: str, wanted: Sequence[str]) -> bool:
    symbol = _spacegroup_symbol(label)
    number = label.split(" ", 1)[0]
    return label in wanted or symbol in wanted or number in wanted


def find_cifs(cif_dir: Path, limit: int) -> List[Path]:
    return sorted(cif_dir.glob("*.cif"))[:limit]


def parse_model_candidates(
    spec: mlip.ModelSpec,
    base_dir: Path,
    symmetry_by_file: Dict[str, str],
    wanted_spacegroups: Sequence[str],
    limit: int,
    recursive_json: bool,
) -> Tuple[List[Candidate], str]:
    cifs = find_cifs(spec.cif_dir, limit)
    json_files, pattern_used = mlip.find_json_files(base_dir, spec, recursive=recursive_json)
    json_files = sorted(json_files, key=mlip._task_number)
    candidates: List[Candidate] = []
    offset = 0
    unique_assumption = "all_valid_due_to_missing_per_structure_uniqueness"

    for json_path in json_files:
        data = json.load(open(json_path))
        run_n = int(data.get("run_info", {}).get("n_structures", 0))
        valid_ids = data.get("validity_filtering", {}).get("valid_structure_ids", [])
        novelty_values = _extract_list(data["results"]["novelty"])
        uniqueness_values = _extract_list(data["results"]["uniqueness"])
        stability_values = mlip._extract_individual_values(data["results"]["stability"], "stability")

        for pos, local_id in enumerate(valid_ids):
            if pos >= len(stability_values) or pos >= len(novelty_values):
                continue
            global_idx = offset + int(local_id)
            if global_idx >= len(cifs):
                continue
            cif_path = cifs[global_idx]
            sg_label = symmetry_by_file.get(str(cif_path))
            if not sg_label or not matches_spacegroup(sg_label, wanted_spacegroups):
                continue
            novelty = float(novelty_values[pos])
            if novelty < 1.0:
                continue
            unique = bool(uniqueness_values[pos]) if pos < len(uniqueness_values) else True
            if not unique:
                continue
            ehull = mlip._as_float(stability_values[pos].get("value"))
            if not np.isfinite(ehull):
                continue
            status = "Stable" if ehull <= 0 else ("Metastable" if ehull <= 0.1 else "Unstable")
            if status not in {"Stable", "Metastable"}:
                continue
            candidates.append(
                Candidate(
                    model=spec.name,
                    global_index=global_idx,
                    local_index=int(local_id),
                    json_file=str(json_path),
                    cif_path=cif_path,
                    spacegroup_label=sg_label,
                    spacegroup_symbol=_spacegroup_symbol(sg_label),
                    status=status,
                    novelty=novelty,
                    unique=unique,
                    mean_e_above_hull=ehull,
                )
            )
        offset += run_n
    return candidates, unique_assumption


def select_candidates(candidates: Sequence[Candidate], per_status_per_spacegroup: int) -> List[Candidate]:
    grouped: Dict[Tuple[str, str, str], List[Candidate]] = defaultdict(list)
    for cand in candidates:
        grouped[(cand.model, cand.status, cand.spacegroup_label)].append(cand)

    selected = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1], item[2])):
        group = sorted(grouped[key], key=lambda cand: (abs(cand.mean_e_above_hull), cand.global_index))
        selected.extend(group[:per_status_per_spacegroup])
    return selected


def element_color(symbol: str):
    z = Element(symbol).Z
    return mpl.colormaps["tab20"]((z % 20) / 19)


def render_structure(cif_path: Path, output: Path, title: str = "") -> None:
    structure = Structure.from_file(str(cif_path), primitive=False, sort=False)
    coords = np.asarray([site.frac_coords for site in structure])
    species = [site.specie.symbol for site in structure]
    sizes = np.asarray([Element(sym).atomic_radius or 1.0 for sym in species], dtype=float)
    sizes = 65 * (sizes / max(sizes.max(), 1e-6)) ** 2 + 20

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    fig.patch.set_facecolor("white")
    for sym in sorted(set(species)):
        idx = [i for i, value in enumerate(species) if value == sym]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=sizes[idx],
            color=element_color(sym),
            edgecolor="#263238",
            linewidth=0.45,
            alpha=0.92,
            label=sym,
        )
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8.5, pad=3)
    for spine in ax.spines.values():
        spine.set_color("#cfd8dc")
        spine.set_linewidth(0.8)
    fig.savefig(output, dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_contact_sheet(image_paths: Sequence[Path], output: Path, title: str) -> None:
    if not image_paths:
        return
    rows, cols = 5, 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.axis("off")
    for ax, path in zip(axes.flat, image_paths[: rows * cols]):
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(pad=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def copy_and_visualize(selected: Sequence[Candidate], output_dir: Path) -> pd.DataFrame:
    records = []
    image_by_model_status: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for cand in selected:
        model_dir = output_dir / _safe_filename(cand.model) / cand.status.lower()
        cif_dir = model_dir / "cifs"
        img_dir = model_dir / "images"
        cif_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{_safe_filename(cand.spacegroup_symbol)}_{cand.global_index:04d}_{_safe_filename(cand.status)}"
        target_cif = cif_dir / f"{stem}.cif"
        target_png = img_dir / f"{stem}.png"
        shutil.copy2(cand.cif_path, target_cif)
        render_structure(
            cand.cif_path,
            target_png,
            title=f"{cand.spacegroup_symbol} | {cand.mean_e_above_hull:.3f} eV/atom",
        )
        image_by_model_status[(cand.model, cand.status)].append(target_png)
        records.append(
            {
                "model": cand.model,
                "status": cand.status,
                "spacegroup_label": cand.spacegroup_label,
                "spacegroup_symbol": cand.spacegroup_symbol,
                "global_index": cand.global_index,
                "local_index": cand.local_index,
                "mean_e_above_hull": cand.mean_e_above_hull,
                "novelty": cand.novelty,
                "unique": cand.unique,
                "source_cif": str(cand.cif_path),
                "copied_cif": str(target_cif),
                "image": str(target_png),
                "json_file": cand.json_file,
            }
        )

    for (model, status), paths in image_by_model_status.items():
        sheet = output_dir / _safe_filename(model) / f"{status.lower()}_contact_sheet_5x3.png"
        make_contact_sheet(paths, sheet, f"{model}: novel/unique/{status.lower()}")

    return pd.DataFrame(records)


def write_coverage_report(
    candidates: Sequence[Candidate],
    selected: Sequence[Candidate],
    models: Sequence[str],
    spacegroups: Sequence[str],
    output: Path,
) -> None:
    candidate_counts = Counter((c.model, c.spacegroup_label, c.status) for c in candidates)
    selected_counts = Counter((c.model, c.spacegroup_label, c.status) for c in selected)
    rows = []
    for model in models:
        for sg_label in spacegroups:
            for status in ["Stable", "Metastable"]:
                rows.append(
                    {
                        "model": model,
                        "spacegroup_label": sg_label,
                        "spacegroup_symbol": _spacegroup_symbol(sg_label),
                        "status": status,
                        "candidate_count": candidate_counts.get((model, sg_label, status), 0),
                        "selected_count": selected_counts.get((model, sg_label, status), 0),
                    }
                )
    pd.DataFrame(rows).to_csv(output, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select representative novel/unique stable/metastable CIFs by Spacegroup.")
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--symmetry-csv", type=Path, default=DEFAULT_SYMMETRY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", action="append", default=None, help="Registered model name. Repeatable.")
    parser.add_argument(
        "--custom-model",
        action="append",
        nargs=3,
        metavar=("NAME", "JSON_GLOB", "CIF_DIR"),
        help="Custom model specification. Repeatable.",
    )
    parser.add_argument("--spacegroup", action="append", help="Spacegroup symbol, number, or full label. Repeatable.")
    parser.add_argument("--top-spacegroups", type=int, default=15, help="Use top N MP20 Spacegroups when --spacegroup is omitted.")
    parser.add_argument("--reference-model", default="MP20", help="Reference model for default top Spacegroups.")
    parser.add_argument("--per-status-per-spacegroup", type=int, default=1)
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--no-recursive-json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    models = args.model or DEFAULT_MODELS
    specs = mlip.load_model_specs("all", only_models=models, custom_models=args.custom_model)
    symmetry_by_file = load_spacegroup_by_file(args.symmetry_csv)
    default_sgs = top_spacegroups_from_reference(args.symmetry_csv, args.reference_model, args.top_spacegroups)
    wanted_spacegroups = normalize_spacegroup_inputs(args.spacegroup, default_sgs)

    all_candidates = []
    all_selected = []
    summary_rows = []
    for spec in specs:
        candidates, unique_assumption = parse_model_candidates(
            spec,
            args.base_dir,
            symmetry_by_file,
            wanted_spacegroups,
            args.limit,
            recursive_json=not args.no_recursive_json,
        )
        selected = select_candidates(candidates, args.per_status_per_spacegroup)
        all_candidates.extend(candidates)
        all_selected.extend(selected)
        summary_rows.append(
            {
                "model": spec.name,
                "candidate_count": len(candidates),
                "selected_count": len(selected),
                "unique_assumption": unique_assumption,
                "requested_spacegroups": ";".join(wanted_spacegroups),
            }
        )
        print(f"{spec.name}: {len(candidates)} candidates, {len(selected)} selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_df = copy_and_visualize(all_selected, args.output_dir)
    selected_df.to_csv(args.output_dir / "selected_representative_structures.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "selection_summary.csv", index=False)
    write_coverage_report(all_candidates, all_selected, [spec.name for spec in specs], wanted_spacegroups, args.output_dir / "selection_coverage_by_spacegroup.csv")
    print(f"Wrote representative structures to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
