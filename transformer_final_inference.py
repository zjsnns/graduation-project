import argparse
import ast
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SCRIPT_DIR = Path(__file__).resolve().parent
SUNDIAL_DIR = SCRIPT_DIR / "sundial"
if str(SUNDIAL_DIR) not in sys.path:
    sys.path.append(str(SUNDIAL_DIR))

from sundial_finetune_utils import load_clean_well_data_with_options  # noqa: E402


BASE_FEATURE_COLS = ["gas_rate", "tubing_pressure", "casing_pressure"]
DIFF_FEATURE_BASE_COL = {
    "gas_rate_diff_rate": "gas_rate",
    "tubing_pressure_diff_rate": "tubing_pressure",
    "casing_pressure_diff_rate": "casing_pressure",
}


class MultiStepTransformerMLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        nhead: int,
        output_dim: int,
        head_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.input_net = nn.Linear(input_dim, hidden_size)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1024, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, output_dim),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.input_net(src)
        src = src + self.pos_encoder[:, : src.size(1), :]
        output = self.transformer_encoder(src)
        return self.output_net(output[:, -1, :])


@dataclass
class InferenceCase:
    well_id: str
    history_norm: np.ndarray
    target_norm: np.ndarray
    pred_norm: np.ndarray
    history_raw: np.ndarray
    target_raw: np.ndarray
    pred_raw: np.ndarray
    mae_norm: float
    rmse_norm: float
    r2_norm: float
    mae_raw: float
    rmse_raw: float
    r2_raw: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final Transformer inference and evaluation script.")
    parser.add_argument(
        "--project_dir",
        type=str,
        default=r"D:\graduation project\Natural Gas Dataset\Natural Gas Dataset",
        help="Directory containing raw well Excel files and the cleaning utilities.",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=str(SCRIPT_DIR / "final_transformer_model"),
        help="Directory containing best_model.pth and run_config.txt.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR / "inference_outputs"))
    parser.add_argument("--plot_top_k", type=int, default=3)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    return parser.parse_args()


def parse_run_config(config_path: Path) -> Dict[str, object]:
    config: Dict[str, object] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            config[key] = ast.literal_eval(value)
        except Exception:
            config[key] = value
    return config


def as_int(config: Dict[str, object], key: str) -> int:
    return int(config[key])


def as_float(config: Dict[str, object], key: str) -> float:
    return float(config[key])


def add_diff_rate_feature(df: pd.DataFrame, base_col: str, new_col: str, eps: float) -> pd.DataFrame:
    df = df.sort_values(["well_id", "date"]).copy()
    prev = df.groupby("well_id")[base_col].shift(1)
    denom = prev.abs().clip(lower=eps)
    diff_rate = (df[base_col] - prev) / denom
    df[new_col] = diff_rate.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return df


def prepare_dataframe(project_dir: Path, config: Dict[str, object]) -> Tuple[pd.DataFrame, List[str]]:
    extra_feature = str(config.get("extra_feature", "none"))
    min_uptime = as_float(config, "min_uptime")
    diff_eps = float(config.get("diff_eps", 1e-6))
    df = load_clean_well_data_with_options(
        data_dir=str(project_dir),
        min_uptime=min_uptime,
        keep_low_uptime=False,
        add_shutin_days=False,
        add_prev_shutin_days=(extra_feature == "prev_shutin_days"),
        add_days_since_open=(extra_feature == "days_since_open"),
        shutin_uptime_threshold=min_uptime,
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["well_id", "date"]).reset_index(drop=True)

    feature_cols = list(config.get("feature_cols", BASE_FEATURE_COLS))
    if extra_feature in DIFF_FEATURE_BASE_COL and extra_feature not in df.columns:
        df = add_diff_rate_feature(df, DIFF_FEATURE_BASE_COL[extra_feature], extra_feature, diff_eps)

    missing = sorted(set(feature_cols) - set(df.columns))
    if missing:
        raise ValueError(f"Missing feature columns after preprocessing: {missing}")
    return df, feature_cols


def build_eval_samples(
    df: pd.DataFrame,
    feature_cols: List[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    seq_len: int,
    pred_len: int,
    split: str,
) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    samples: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    min_required = seq_len + pred_len * 3
    gas_idx = feature_cols.index("gas_rate")

    for well_id, grp in df.groupby("well_id"):
        grp = grp.sort_values("date")
        raw_values = grp[feature_cols].values.astype(np.float32)
        n = len(raw_values)
        if n < min_required:
            continue
        norm_values = (raw_values - feature_means) / feature_stds

        if split == "val":
            history = norm_values[n - pred_len * 2 - seq_len : n - pred_len * 2]
            target = norm_values[n - pred_len * 2 : n - pred_len, gas_idx]
            raw_window = raw_values[n - pred_len * 2 - seq_len : n - pred_len, gas_idx]
        else:
            history = norm_values[n - pred_len - seq_len : n - pred_len]
            target = norm_values[n - pred_len :, gas_idx]
            raw_window = raw_values[n - pred_len - seq_len :, gas_idx]

        samples.append(
            (
                str(well_id),
                history.astype(np.float32),
                target.astype(np.float32),
                raw_window.astype(np.float32),
            )
        )
    return samples


def evaluate(
    model: nn.Module,
    samples: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    device: torch.device,
    gas_mean: float,
    gas_std: float,
    seq_len: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[InferenceCase]]:
    rows = []
    point_rows = []
    cases: List[InferenceCase] = []
    all_true_norm = []
    all_pred_norm = []
    all_true_raw = []
    all_pred_raw = []

    model.eval()
    with torch.no_grad():
        for well_id, history, target, raw_window in samples:
            x = torch.from_numpy(history).unsqueeze(0).to(device=device, dtype=torch.float32)
            pred = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)

            history_raw = raw_window[:seq_len]
            target_raw = target * gas_std + gas_mean
            pred_raw = pred * gas_std + gas_mean

            mae_norm = float(mean_absolute_error(target, pred))
            rmse_norm = float(math.sqrt(mean_squared_error(target, pred)))
            r2_norm = float(r2_score(target, pred))
            mae_raw = float(mean_absolute_error(target_raw, pred_raw))
            rmse_raw = float(math.sqrt(mean_squared_error(target_raw, pred_raw)))
            r2_raw = float(r2_score(target_raw, pred_raw))

            rows.append(
                {
                    "well_id": well_id,
                    "mae_norm": mae_norm,
                    "rmse_norm": rmse_norm,
                    "r2_norm": r2_norm,
                    "mae_raw": mae_raw,
                    "rmse_raw": rmse_raw,
                    "r2_raw": r2_raw,
                }
            )
            for step, (y_true, y_pred, y_true_raw, y_pred_raw) in enumerate(
                zip(target, pred, target_raw, pred_raw), start=1
            ):
                point_rows.append(
                    {
                        "well_id": well_id,
                        "step": step,
                        "true_norm": float(y_true),
                        "pred_norm": float(y_pred),
                        "true_raw": float(y_true_raw),
                        "pred_raw": float(y_pred_raw),
                    }
                )

            cases.append(
                InferenceCase(
                    well_id=well_id,
                    history_norm=history[:, 0].copy(),
                    target_norm=target.copy(),
                    pred_norm=pred.copy(),
                    history_raw=history_raw.copy(),
                    target_raw=target_raw.copy(),
                    pred_raw=pred_raw.copy(),
                    mae_norm=mae_norm,
                    rmse_norm=rmse_norm,
                    r2_norm=r2_norm,
                    mae_raw=mae_raw,
                    rmse_raw=rmse_raw,
                    r2_raw=r2_raw,
                )
            )
            all_true_norm.append(target)
            all_pred_norm.append(pred)
            all_true_raw.append(target_raw)
            all_pred_raw.append(pred_raw)

    per_well_df = pd.DataFrame(rows).sort_values("r2_norm", ascending=False).reset_index(drop=True)
    points_df = pd.DataFrame(point_rows)
    pooled_true_norm = np.concatenate(all_true_norm)
    pooled_pred_norm = np.concatenate(all_pred_norm)
    pooled_true_raw = np.concatenate(all_true_raw)
    pooled_pred_raw = np.concatenate(all_pred_raw)
    summary_df = pd.DataFrame(
        [
            {
                "num_wells": len(samples),
                "mean_mae_norm": float(per_well_df["mae_norm"].mean()),
                "mean_rmse_norm": float(per_well_df["rmse_norm"].mean()),
                "mean_r2_norm": float(per_well_df["r2_norm"].mean()),
                "pooled_mae_norm": float(mean_absolute_error(pooled_true_norm, pooled_pred_norm)),
                "pooled_rmse_norm": float(math.sqrt(mean_squared_error(pooled_true_norm, pooled_pred_norm))),
                "pooled_r2_norm": float(r2_score(pooled_true_norm, pooled_pred_norm)),
                "mean_mae_raw": float(per_well_df["mae_raw"].mean()),
                "mean_rmse_raw": float(per_well_df["rmse_raw"].mean()),
                "mean_r2_raw": float(per_well_df["r2_raw"].mean()),
                "pooled_mae_raw": float(mean_absolute_error(pooled_true_raw, pooled_pred_raw)),
                "pooled_rmse_raw": float(math.sqrt(mean_squared_error(pooled_true_raw, pooled_pred_raw))),
                "pooled_r2_raw": float(r2_score(pooled_true_raw, pooled_pred_raw)),
            }
        ]
    )
    return per_well_df, summary_df, points_df, cases


def save_case_plot(case: InferenceCase, seq_len: int, pred_len: int, save_path: Path, title: str) -> None:
    x_hist = np.arange(seq_len)
    x_future = np.arange(seq_len, seq_len + pred_len)
    plt.figure(figsize=(11, 4.8))
    plt.plot(x_hist, case.history_raw, label="History", color="#222222", linewidth=1.7)
    plt.plot(x_future, case.target_raw, label="True", color="#1b7f3a", linewidth=1.8)
    plt.plot(x_future, case.pred_raw, label="Predicted", color="#c23b22", linewidth=1.8, linestyle="--")
    plt.axvline(seq_len - 1, color="#666666", linewidth=1.0, linestyle=":")
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Gas rate")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    config = parse_run_config(run_dir / "run_config.txt")
    seq_len = as_int(config, "seq_len")
    pred_len = as_int(config, "pred_len")
    feature_means = np.asarray(config["feature_means"], dtype=np.float32)
    feature_stds = np.asarray(config["feature_stds"], dtype=np.float32)

    df, feature_cols = prepare_dataframe(project_dir, config)
    if feature_cols != list(config["feature_cols"]):
        raise ValueError(f"Feature column mismatch: {feature_cols} != {config['feature_cols']}")

    samples = build_eval_samples(
        df=df,
        feature_cols=feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        seq_len=seq_len,
        pred_len=pred_len,
        split=args.split,
    )
    if not samples:
        raise RuntimeError("No evaluation samples were created.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MultiStepTransformerMLPHead(
        input_dim=len(feature_cols),
        hidden_size=as_int(config, "hidden_size"),
        num_layers=as_int(config, "num_layers"),
        nhead=as_int(config, "nhead"),
        output_dim=pred_len,
        head_hidden_dim=as_int(config, "head_hidden_dim"),
        dropout=as_float(config, "dropout"),
    ).to(device)
    state = torch.load(run_dir / "best_model.pth", map_location=device)
    model.load_state_dict(state)

    gas_idx = feature_cols.index("gas_rate")
    per_well_df, summary_df, points_df, cases = evaluate(
        model=model,
        samples=samples,
        device=device,
        gas_mean=float(feature_means[gas_idx]),
        gas_std=float(feature_stds[gas_idx]),
        seq_len=seq_len,
    )

    per_well_df.to_csv(output_dir / f"{args.split}_per_well_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / f"{args.split}_summary_metrics.csv", index=False, encoding="utf-8-sig")
    points_df.to_csv(output_dir / f"{args.split}_prediction_points.csv", index=False, encoding="utf-8-sig")

    sorted_cases = sorted(cases, key=lambda c: c.r2_norm, reverse=True)
    top_k = min(args.plot_top_k, len(sorted_cases))
    for idx, case in enumerate(sorted_cases[:top_k], start=1):
        save_case_plot(
            case=case,
            seq_len=seq_len,
            pred_len=pred_len,
            save_path=plot_dir / f"{args.split}_best_{idx}_{case.well_id}_r2_{case.r2_norm:.4f}.png",
            title=f"{args.split.upper()} best #{idx} | Well {case.well_id} | R2={case.r2_norm:.4f}",
        )
    for idx, case in enumerate(sorted_cases[-top_k:], start=1):
        save_case_plot(
            case=case,
            seq_len=seq_len,
            pred_len=pred_len,
            save_path=plot_dir / f"{args.split}_worst_{idx}_{case.well_id}_r2_{case.r2_norm:.4f}.png",
            title=f"{args.split.upper()} worst #{idx} | Well {case.well_id} | R2={case.r2_norm:.4f}",
        )

    row = summary_df.iloc[0].to_dict()
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Split: {args.split} | wells={int(row['num_wells'])}")
    print(
        "[INFO] Normalized metrics: "
        f"mean_MAE={row['mean_mae_norm']:.6f}, "
        f"mean_RMSE={row['mean_rmse_norm']:.6f}, "
        f"mean_R2={row['mean_r2_norm']:.6f}, "
        f"pooled_R2={row['pooled_r2_norm']:.6f}"
    )
    print(
        "[INFO] Raw-scale metrics: "
        f"mean_MAE={row['mean_mae_raw']:.6f}, "
        f"mean_RMSE={row['mean_rmse_raw']:.6f}, "
        f"mean_R2={row['mean_r2_raw']:.6f}, "
        f"pooled_R2={row['pooled_r2_raw']:.6f}"
    )
    print(f"[INFO] Saved outputs: {output_dir}")


if __name__ == "__main__":
    main()
