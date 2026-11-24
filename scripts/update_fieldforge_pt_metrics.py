#!/usr/bin/env python3
"""Replace only FieldForge (PT) MLIP metrics in an existing analysis output.

The structural/CIF analyses are intentionally outside this workflow.  The
replacement JSON is parsed with the same rules as the main metric script,
while every non-FieldForge row already present in the output directory is
kept verbatim at the dataframe level.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import analyze_mlip_relaxation_metrics as metrics
from settings import FIELD_FORGE_CIF_DIR, FIELD_FORGE_NAME, OUTPUT_ROOT


MODEL = FIELD_FORGE_NAME
DEFAULT_JSON = metrics.BASE_DIR / "fieldforge_metrics.json"
DEFAULT_OUTPUT = OUTPUT_ROOT / "mlip_metrics"
DEFAULT_CIF_DIR = FIELD_FORGE_CIF_DIR


def _read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.exists():
        result = pd.read_csv(path)
        if "metric" in result.columns:
            result["metric"] = result["metric"].replace(
                {"relaxation_RMSE": "relaxation_RMSD"}
            )
        return result
    return pd.DataFrame(columns=columns or [])


def _without_model(df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in df:
        return df.copy()
    return df[df["model"] != MODEL].copy()


def _concat(*frames: pd.DataFrame) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return frames[0].copy() if frames else pd.DataFrame()
    return pd.concat(nonempty, ignore_index=True)


def _order_by_model(df: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    if df.empty or "model" not in df:
        return df
    order = {model: idx for idx, model in enumerate(model_order)}
    result = df.copy()
    result["_model_order"] = result["model"].map(order).fillna(len(order))
    result = result.sort_values("_model_order", kind="stable").drop(columns="_model_order")
    return result.reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replace only FieldForge (PT) in an existing MLIP metric analysis."
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-dir", type=Path, default=metrics.BASE_DIR)
    parser.add_argument("--element-cache", type=Path, default=metrics.ELEMENT_CACHE)
    parser.add_argument("--cif-dir", type=Path, default=DEFAULT_CIF_DIR)
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--outlier-iqr-multiplier", type=float, default=3.0)
    parser.add_argument("--dpi", type=int, default=600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.json.exists():
        raise SystemExit(f"Replacement JSON not found: {args.json}")
    if not args.output_dir.exists():
        raise SystemExit(f"Existing analysis output directory not found: {args.output_dir}")

    metrics._configure_style()
    spec = metrics.ModelSpec(MODEL, args.json.name, args.cif_dir)
    new_raw, new_task, new_source = metrics.parse_all_metrics(
        [spec],
        args.base_dir,
        args.element_cache,
        args.limit,
        recursive_json=True,
    )
    new_filtered, new_outliers = metrics.split_metric_outliers(
        new_raw, args.outlier_iqr_multiplier
    )

    csv_dir = args.output_dir / "csv"
    old_raw = _read_csv(csv_dir / "per_structure_metric_values_long_raw.csv")
    old_filtered = _read_csv(csv_dir / "per_structure_metric_values_long.csv")
    old_task = _read_csv(csv_dir / "task_metric_summary_from_json.csv")
    old_source = _read_csv(csv_dir / "json_sources.csv")
    old_outliers = _read_csv(csv_dir / "outliers" / "all_metric_outliers.csv")

    raw = _concat(_without_model(old_raw), new_raw)
    filtered = _concat(_without_model(old_filtered), new_filtered)
    task = _concat(_without_model(old_task), new_task)
    source = _concat(_without_model(old_source), new_source)
    outliers = _concat(_without_model(old_outliers), new_outliers)

    model_order = [MODEL]
    if not source.empty:
        model_order.extend(
            model for model in source["model"].drop_duplicates().tolist() if model != MODEL
        )
    raw = _order_by_model(raw, model_order)
    filtered = _order_by_model(filtered, model_order)
    task = _order_by_model(task, model_order)
    source = _order_by_model(source, model_order)
    outliers = _order_by_model(outliers, model_order)
    model_summary = metrics.apply_json_mean_values(
        metrics.aggregate_model_summary(filtered), task
    )
    sm_df = metrics.stable_metastable_distribution(filtered)

    # Remove only the previous FieldForge per-model outlier file.  The common
    # writer then recreates it from the replacement JSON and leaves all other
    # model outlier files untouched.
    old_pt_outlier = csv_dir / "outliers" / f"{metrics._safe_filename(MODEL)}_metric_outliers.csv"
    if old_pt_outlier.exists():
        old_pt_outlier.unlink()

    metrics.write_outputs(
        raw,
        filtered,
        task,
        model_summary,
        source,
        sm_df,
        outliers,
        args.output_dir,
    )

    # Only this per-model plot is regenerated.  The combined plots necessarily
    # use the merged dataframe, but all non-FieldForge input rows are retained.
    metrics.plot_per_model_metric_distributions(
        new_filtered, [MODEL], args.output_dir, args.dpi
    )
    metrics.plot_mean_metric_comparison(filtered, model_order, args.output_dir, args.dpi)
    mp20_mean = metrics.load_mp20_formation_energy_mean(metrics.MP20_CSV)
    metrics.plot_model_metric_summary_bars(
        model_summary, model_order, args.output_dir, args.dpi, mp20_mean
    )
    metrics.plot_stable_metastable(
        sm_df,
        model_order,
        args.output_dir,
        args.dpi,
        per_model_order=[MODEL],
    )

    print(f"Updated {MODEL} from: {args.json}")
    print(f"Replacement JSON: {len(new_raw):,} raw metric values")
    print(f"After IQR filtering: {len(new_filtered):,} values")
    print(f"Dropped outliers: {len(new_outliers):,}")
    print(f"Wrote updated outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
