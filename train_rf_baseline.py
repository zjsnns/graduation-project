import argparse
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
SUNDIAL_DIR = SCRIPT_DIR / "sundial"
if str(SUNDIAL_DIR) not in sys.path:
    sys.path.append(str(SUNDIAL_DIR))

from sundial_finetune_utils import load_clean_well_data_with_options  # noqa: E402


BASE_FEATURE_COLS = ["gas_rate", "tubing_pressure", "casing_pressure"]


def default_data_dir() -> Path:
    candidates = [
        SCRIPT_DIR.parent.parent / "Natural Gas Dataset",
        SCRIPT_DIR,
    ]
    for path in candidates:
        if path.exists() and any(path.glob("*.xlsx")):
            return path
    return SCRIPT_DIR


@dataclass
class EvalCase:
    well_id: str
    true_future: np.ndarray
    pred_future: np.ndarray
    mae: float
    rmse: float
    r2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict dynamic-cleaning RandomForest baseline (same split/normalization protocol)."
    )
    parser.add_argument("--data_dir", type=str, default=str(default_data_dir()))
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--min_uptime", type=float, default=20.0)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=18)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_root", type=str, default=str(SCRIPT_DIR / "rf_runs_strict_baseline"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def prepare_dataframe(data_dir: Path, min_uptime: float) -> pd.DataFrame:
    df = load_clean_well_data_with_options(
        data_dir=str(data_dir),
        min_uptime=min_uptime,
        keep_low_uptime=False,
        add_shutin_days=False,
        add_prev_shutin_days=False,
        add_days_since_open=False,
        shutin_uptime_threshold=min_uptime,
    )
    required = {"date", "well_id", *BASE_FEATURE_COLS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values(["well_id", "date"]).reset_index(drop=True)
    return df


def fit_feature_stats(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    pred_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    min_required = seq_len + pred_len * 3
    train_parts = []
    for _, grp in df.groupby("well_id"):
        grp = grp.sort_values("date")
        if len(grp) < min_required:
            continue
        train_end = len(grp) - pred_len * 2
        train_parts.append(grp.iloc[:train_end][feature_cols].values.astype(np.float32))
    if not train_parts:
        raise RuntimeError("No eligible wells for feature fitting.")
    stacked = np.concatenate(train_parts, axis=0)
    means = stacked.mean(axis=0).astype(np.float32)
    stds = stacked.std(axis=0).astype(np.float32)
    stds = np.where(stds > 1e-6, stds, 1.0).astype(np.float32)
    return means, stds


def build_time_split_data(
    df: pd.DataFrame,
    seq_len: int,
    pred_len: int,
    stride: int,
    feature_cols: List[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, np.ndarray, np.ndarray]], List[Tuple[str, np.ndarray, np.ndarray]], List[Dict[str, int]]]:
    train_x = []
    train_y = []
    val_samples: List[Tuple[str, np.ndarray, np.ndarray]] = []
    test_samples: List[Tuple[str, np.ndarray, np.ndarray]] = []
    well_stats: List[Dict[str, int]] = []

    min_required = seq_len + pred_len * 3
    gas_idx = feature_cols.index("gas_rate")

    for well_id, grp in df.groupby("well_id"):
        grp = grp.sort_values("date")
        raw_values = grp[feature_cols].values.astype(np.float32)
        n = len(raw_values)
        if n < min_required:
            continue

        values = (raw_values - feature_means) / feature_stds
        train_end = n - pred_len * 2
        train_values = values[:train_end]
        train_window_count = 0
        max_start = len(train_values) - seq_len - pred_len
        for start in range(0, max_start + 1, stride):
            history = train_values[start : start + seq_len]
            target = train_values[start + seq_len : start + seq_len + pred_len, gas_idx]
            train_x.append(history.reshape(-1).astype(np.float32))
            train_y.append(target.astype(np.float32))
            train_window_count += 1

        if train_window_count == 0:
            continue

        val_history = values[n - pred_len * 2 - seq_len : n - pred_len * 2]
        val_target = values[n - pred_len * 2 : n - pred_len, gas_idx]
        test_history = values[n - pred_len - seq_len : n - pred_len]
        test_target = values[n - pred_len :, gas_idx]

        val_samples.append((str(well_id), val_history.astype(np.float32), val_target.astype(np.float32)))
        test_samples.append((str(well_id), test_history.astype(np.float32), test_target.astype(np.float32)))
        well_stats.append(
            {
                "well_id": str(well_id),
                "points": n,
                "train_windows": train_window_count,
            }
        )

    if not train_x:
        raise RuntimeError("No train samples were created.")
    if not val_samples or not test_samples:
        raise RuntimeError("Validation or test samples are empty.")
    return np.asarray(train_x), np.asarray(train_y), val_samples, test_samples, well_stats


def evaluate_cases(
    model: RandomForestRegressor,
    samples: List[Tuple[str, np.ndarray, np.ndarray]],
) -> Tuple[pd.DataFrame, List[EvalCase], Dict[str, float]]:
    rows = []
    cases: List[EvalCase] = []
    all_true = []
    all_pred = []

    for well_id, history, target in samples:
        pred = model.predict(history.reshape(1, -1)).reshape(-1).astype(np.float32)
        sample_mae = float(mean_absolute_error(target, pred))
        sample_rmse = float(math.sqrt(mean_squared_error(target, pred)))
        sample_r2 = float(r2_score(target, pred))
        rows.append({"well_id": well_id, "mae": sample_mae, "rmse": sample_rmse, "r2": sample_r2})
        cases.append(
            EvalCase(
                well_id=well_id,
                true_future=target.copy(),
                pred_future=pred.copy(),
                mae=sample_mae,
                rmse=sample_rmse,
                r2=sample_r2,
            )
        )
        all_true.append(target)
        all_pred.append(pred)

    per_well_df = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    pooled_true = np.concatenate(all_true, axis=0)
    pooled_pred = np.concatenate(all_pred, axis=0)
    summary = {
        "num_wells": len(samples),
        "mean_mae": float(per_well_df["mae"].mean()),
        "mean_rmse": float(per_well_df["rmse"].mean()),
        "mean_r2": float(per_well_df["r2"].mean()),
        "pooled_mae": float(mean_absolute_error(pooled_true, pooled_pred)),
        "pooled_rmse": float(math.sqrt(mean_squared_error(pooled_true, pooled_pred))),
        "pooled_r2": float(r2_score(pooled_true, pooled_pred)),
    }
    return per_well_df, cases, summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = list(BASE_FEATURE_COLS)
    df = prepare_dataframe(data_dir=data_dir, min_uptime=args.min_uptime)
    feature_means, feature_stds = fit_feature_stats(
        df=df, feature_cols=feature_cols, seq_len=args.seq_len, pred_len=args.pred_len
    )
    train_x, train_y, val_samples, test_samples, well_stats = build_time_split_data(
        df=df,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride,
        feature_cols=feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)

    val_df, _, val_summary = evaluate_cases(model, val_samples)
    test_df, _, test_summary = evaluate_cases(model, test_samples)

    pd.DataFrame(well_stats).sort_values("train_windows", ascending=False).to_csv(run_dir / "well_stats.csv", index=False)
    val_df.to_csv(run_dir / "val_per_well_metrics.csv", index=False)
    test_df.to_csv(run_dir / "test_per_well_metrics.csv", index=False)
    pd.DataFrame([{"split": "val", **val_summary}, {"split": "test", **test_summary}]).to_csv(
        run_dir / "summary_metrics.csv", index=False
    )
    (run_dir / "run_config.txt").write_text(
        "\n".join(
            [
                f"baseline_name=strict_random_forest_baseline",
                f"seed={args.seed}",
                f"seq_len={args.seq_len}",
                f"pred_len={args.pred_len}",
                f"stride={args.stride}",
                f"min_uptime={args.min_uptime}",
                f"n_estimators={args.n_estimators}",
                f"max_depth={args.max_depth}",
                f"min_samples_leaf={args.min_samples_leaf}",
                f"feature_cols={feature_cols}",
                f"feature_means={feature_means.tolist()}",
                f"feature_stds={feature_stds.tolist()}",
                f"eligible_wells={len(well_stats)}",
                f"train_windows={len(train_x)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n========== STRICT RANDOM-FOREST BASELINE SUMMARY ==========")
    print(
        f"VAL  | wells={val_summary['num_wells']} "
        f"mean_mae={val_summary['mean_mae']:.6f} "
        f"mean_rmse={val_summary['mean_rmse']:.6f} "
        f"mean_r2={val_summary['mean_r2']:.6f} "
        f"pooled_r2={val_summary['pooled_r2']:.6f}"
    )
    print(
        f"TEST | wells={test_summary['num_wells']} "
        f"mean_mae={test_summary['mean_mae']:.6f} "
        f"mean_rmse={test_summary['mean_rmse']:.6f} "
        f"mean_r2={test_summary['mean_r2']:.6f} "
        f"pooled_r2={test_summary['pooled_r2']:.6f}"
    )
    print("=========================================================\n")
    print(f"[INFO] Saved run directory: {run_dir}")


if __name__ == "__main__":
    main()
