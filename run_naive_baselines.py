from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SCRIPT_DIR = Path(__file__).resolve().parent
SUNDIAL_DIR = SCRIPT_DIR / "sundial"
if str(SUNDIAL_DIR) not in sys.path:
    sys.path.append(str(SUNDIAL_DIR))

from sundial_finetune_utils import load_clean_well_data_with_options  # noqa: E402


def default_data_dir() -> Path:
    candidates = [
        SCRIPT_DIR.parent.parent / "Natural Gas Dataset",
        SCRIPT_DIR,
    ]
    for path in candidates:
        if path.exists() and any(path.glob("*.xlsx")):
            return path
    return SCRIPT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict time-split naive baselines for 256->96 shale gas forecasting."
    )
    parser.add_argument("--data_dir", type=str, default=str(default_data_dir()))
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--min_uptime", type=float, default=20.0)
    parser.add_argument("--output_root", type=str, default=str(SCRIPT_DIR / "reports" / "final_report"))
    return parser.parse_args()


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
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["gas_rate"] = pd.to_numeric(df["gas_rate"], errors="coerce")
    df = df.dropna(subset=["date", "gas_rate"]).copy()
    return df.sort_values(["well_id", "date"]).reset_index(drop=True)


def fit_train_gas_stats(df: pd.DataFrame, seq_len: int, pred_len: int) -> Tuple[float, float, List[str]]:
    min_required = seq_len + pred_len * 3
    parts: List[np.ndarray] = []
    eligible_wells: List[str] = []
    for well_id, grp in df.groupby("well_id"):
        grp = grp.sort_values("date")
        if len(grp) < min_required:
            continue
        train_end = len(grp) - pred_len * 2
        parts.append(grp.iloc[:train_end]["gas_rate"].to_numpy(dtype=np.float32))
        eligible_wells.append(str(well_id))
    if not parts:
        raise RuntimeError("No eligible wells after strict split filtering.")
    values = np.concatenate(parts)
    mean = float(values.mean())
    std = float(values.std())
    if std <= 1e-6:
        std = 1.0
    return mean, std, eligible_wells


def build_eval_samples(
    df: pd.DataFrame,
    seq_len: int,
    pred_len: int,
    gas_mean: float,
    gas_std: float,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]], pd.DataFrame]:
    min_required = seq_len + pred_len * 3
    val_samples: List[Dict[str, np.ndarray]] = []
    test_samples: List[Dict[str, np.ndarray]] = []
    stats: List[Dict[str, int]] = []

    for well_id, grp in df.groupby("well_id"):
        grp = grp.sort_values("date")
        raw = grp["gas_rate"].to_numpy(dtype=np.float32)
        n = len(raw)
        if n < min_required:
            continue
        values = ((raw - gas_mean) / gas_std).astype(np.float32)

        val_history = values[n - pred_len * 2 - seq_len : n - pred_len * 2]
        val_target = values[n - pred_len * 2 : n - pred_len]
        test_history = values[n - pred_len - seq_len : n - pred_len]
        test_target = values[n - pred_len :]

        val_samples.append({"well_id": str(well_id), "history": val_history, "target": val_target})
        test_samples.append({"well_id": str(well_id), "history": test_history, "target": test_target})
        stats.append(
            {
                "well_id": str(well_id),
                "points": int(n),
                "train_points": int(n - pred_len * 2),
                "val_points": int(pred_len),
                "test_points": int(pred_len),
            }
        )

    return val_samples, test_samples, pd.DataFrame(stats)


def predict_from_history(history: np.ndarray, pred_len: int, method: str) -> np.ndarray:
    if method == "last_value":
        value = float(history[-1])
    elif method == "ma7":
        value = float(history[-7:].mean())
    elif method == "ma30":
        value = float(history[-30:].mean())
    else:
        raise ValueError(f"Unknown method: {method}")
    return np.full(pred_len, value, dtype=np.float32)


def evaluate_method(
    samples: List[Dict[str, np.ndarray]],
    method: str,
    pred_len: int,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    rows = []
    point_rows = []
    pooled_true = []
    pooled_pred = []

    for sample in samples:
        well_id = str(sample["well_id"])
        target = sample["target"].astype(np.float32)
        pred = predict_from_history(sample["history"], pred_len=pred_len, method=method)
        mae = float(mean_absolute_error(target, pred))
        rmse = float(math.sqrt(mean_squared_error(target, pred)))
        r2 = float(r2_score(target, pred))
        rows.append({"well_id": well_id, "mae": mae, "rmse": rmse, "r2": r2})
        pooled_true.append(target)
        pooled_pred.append(pred)
        for step, (y, yhat) in enumerate(zip(target, pred), start=1):
            point_rows.append({"well_id": well_id, "step": step, "y_true": float(y), "y_pred": float(yhat)})

    per_well = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    y_true_all = np.concatenate(pooled_true)
    y_pred_all = np.concatenate(pooled_pred)
    summary = {
        "method": method,
        "num_wells": int(len(per_well)),
        "mean_mae": float(per_well["mae"].mean()),
        "mean_rmse": float(per_well["rmse"].mean()),
        "mean_r2": float(per_well["r2"].mean()),
        "positive_r2_ratio": float((per_well["r2"] > 0).mean()),
        "pooled_mae": float(mean_absolute_error(y_true_all, y_pred_all)),
        "pooled_rmse": float(math.sqrt(mean_squared_error(y_true_all, y_pred_all))),
        "pooled_r2": float(r2_score(y_true_all, y_pred_all)),
    }
    return per_well, summary, pd.DataFrame(point_rows)


def plot_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.size": 10})
    metrics = [
        ("mean_mae", "Mean MAE", False),
        ("mean_rmse", "Mean RMSE", False),
        ("mean_r2", "Per-well mean R2", True),
        ("pooled_r2", "Pooled R2", True),
        ("positive_r2_ratio", "Positive R2 ratio", True),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 3.2), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for ax, (col, title, hline0) in zip(axes, metrics):
        ax.bar(summary_df["method"], summary_df[col], color=colors[: len(summary_df)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        if hline0:
            ax.axhline(0, color="#333333", linewidth=0.8)
        for i, value in enumerate(summary_df[col]):
            ax.text(i, value, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    fig.suptitle("Naive Baseline Results (strict 256 -> 96 time split)", fontsize=12)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_report(
    out_path: Path,
    summary_df: pd.DataFrame,
    seq_len: int,
    pred_len: int,
    min_uptime: float,
    gas_mean: float,
    gas_std: float,
    num_wells: int,
) -> None:
    best = summary_df.sort_values("mean_rmse").iloc[0]
    lines = [
        "# Naive Baseline Experiment Report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Task: lookback={seq_len}, forecast={pred_len}, strict per-well time split",
        f"- Cleaning: uptime > {min_uptime}",
        f"- Eligible wells: {num_wells}",
        f"- Metrics are computed on standardized gas_rate using training-period mean/std: mean={gas_mean:.6f}, std={gas_std:.6f}",
        "",
        "## Methods",
        "",
        "- last_value: all future steps equal the last gas_rate in the input window.",
        "- ma7: all future steps equal the mean gas_rate of the last 7 input days.",
        "- ma30: all future steps equal the mean gas_rate of the last 30 input days.",
        "",
        "## Test Summary",
        "",
        summary_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Main Finding",
        "",
        (
            f"Among the three naive baselines, `{best['method']}` has the lowest mean RMSE "
            f"({best['mean_rmse']:.6f}). These baselines should be reported before complex models, "
            "because they quantify how much benefit is obtained beyond simple production carry-forward "
            "or short-window averaging."
        ),
        "",
        "## Thesis-ready Chinese Description",
        "",
        (
            "为检验深度模型是否真正超过简单工程外推，本文补充 Last Value、MA7 和 MA30 三种朴素基线。"
            "三种方法均不进行参数训练，只根据输入窗口末端或近期平均值外推未来 96 天日产气量。"
            f"在严格时间切分和相同标准化口径下，三种基线中 {best['method']} 的 mean RMSE 最低，"
            "该结果可作为判断 Transformer、RandomForest、CNN、LSTM 和 Sundial 是否具有实际增益的最低参照。"
        ),
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_dir = output_root / f"naive_baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(data_dir=data_dir, min_uptime=args.min_uptime)
    gas_mean, gas_std, eligible_wells = fit_train_gas_stats(df, seq_len=args.seq_len, pred_len=args.pred_len)
    val_samples, test_samples, well_stats = build_eval_samples(
        df=df,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        gas_mean=gas_mean,
        gas_std=gas_std,
    )

    methods = ["last_value", "ma7", "ma30"]
    val_summaries = []
    test_summaries = []
    for method in methods:
        val_per_well, val_summary, val_points = evaluate_method(val_samples, method, args.pred_len)
        test_per_well, test_summary, test_points = evaluate_method(test_samples, method, args.pred_len)
        val_summary["split"] = "val"
        test_summary["split"] = "test"
        val_summaries.append(val_summary)
        test_summaries.append(test_summary)

        val_per_well.to_csv(output_dir / f"val_per_well_{method}.csv", index=False)
        test_per_well.to_csv(output_dir / f"test_per_well_{method}.csv", index=False)
        val_points.to_csv(output_dir / f"val_points_{method}.csv", index=False)
        test_points.to_csv(output_dir / f"test_points_{method}.csv", index=False)

    well_stats.to_csv(output_dir / "eligible_well_stats.csv", index=False)
    summary_df = pd.DataFrame(val_summaries + test_summaries)
    summary_df = summary_df[
        [
            "split",
            "method",
            "num_wells",
            "mean_mae",
            "mean_rmse",
            "mean_r2",
            "positive_r2_ratio",
            "pooled_mae",
            "pooled_rmse",
            "pooled_r2",
        ]
    ]
    summary_df.to_csv(output_dir / "naive_baseline_summary.csv", index=False)
    test_summary_df = summary_df[summary_df["split"] == "test"].copy()
    plot_summary(test_summary_df, output_dir / "naive_baseline_test_summary.png")
    write_report(
        out_path=output_dir / "naive_baseline_report.md",
        summary_df=test_summary_df,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        min_uptime=args.min_uptime,
        gas_mean=gas_mean,
        gas_std=gas_std,
        num_wells=len(eligible_wells),
    )

    print(f"output_dir={output_dir}")
    print(test_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
