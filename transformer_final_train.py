import argparse
import copy
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
SUNDIAL_DIR = SCRIPT_DIR / "sundial"
if str(SUNDIAL_DIR) not in sys.path:
    sys.path.append(str(SUNDIAL_DIR))

from sundial_finetune_utils import load_clean_well_data_with_options  # noqa: E402


BASE_FEATURE_COLS = ["gas_rate", "tubing_pressure", "casing_pressure"]
EXTRA_FEATURE_CHOICES = [
    "none",
    "prev_shutin_days",
    "days_since_open",
    "gas_rate_diff_rate",
    "tubing_pressure_diff_rate",
    "casing_pressure_diff_rate",
]
DIFF_FEATURE_BASE_COL = {
    "gas_rate_diff_rate": "gas_rate",
    "tubing_pressure_diff_rate": "tubing_pressure",
    "casing_pressure_diff_rate": "casing_pressure",
}


def default_data_dir() -> Path:
    candidates = [
        SCRIPT_DIR.parent.parent / "Natural Gas Dataset",
        SCRIPT_DIR,
    ]
    for path in candidates:
        if path.exists() and any(path.glob("*.xlsx")):
            return path
    return SCRIPT_DIR


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


class WindowDataset(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


@dataclass
class EvalCase:
    well_id: str
    history: np.ndarray
    true_future: np.ndarray
    pred_future: np.ndarray
    mae: float
    rmse: float
    r2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict stable Transformer baseline v2 plus exactly one extra feature."
    )
    parser.add_argument("--data_dir", type=str, default=str(default_data_dir()))
    parser.add_argument("--extra_feature", type=str, default="none", choices=EXTRA_FEATURE_CHOICES)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--head_hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min_uptime", type=float, default=20.0)
    parser.add_argument("--diff_eps", type=float, default=1e-6)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--scheduler_patience", type=int, default=4)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-6)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--min_epochs_before_stop", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--plot_top_k", type=int, default=2)
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--output_root", type=str, default=str(SCRIPT_DIR / "transformer_runs_strict_stable_single_feature"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_diff_rate_feature(df: pd.DataFrame, base_col: str, new_col: str, eps: float) -> pd.DataFrame:
    df = df.sort_values(["well_id", "date"]).copy()
    prev = df.groupby("well_id")[base_col].shift(1)
    denom = prev.abs().clip(lower=eps)
    diff_rate = (df[base_col] - prev) / denom
    diff_rate = diff_rate.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df[new_col] = diff_rate.astype(np.float32)
    return df


def prepare_dataframe(
    data_dir: Path,
    min_uptime: float,
    diff_eps: float,
    extra_feature: str,
) -> Tuple[pd.DataFrame, List[str]]:
    df = load_clean_well_data_with_options(
        data_dir=str(data_dir),
        min_uptime=min_uptime,
        keep_low_uptime=False,
        add_shutin_days=False,
        add_prev_shutin_days=(extra_feature == "prev_shutin_days"),
        add_days_since_open=(extra_feature == "days_since_open"),
        shutin_uptime_threshold=min_uptime,
    )
    required = {"date", "well_id", *BASE_FEATURE_COLS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values(["well_id", "date"]).reset_index(drop=True)

    feature_cols = list(BASE_FEATURE_COLS)
    if extra_feature == "none":
        pass
    elif extra_feature in {"prev_shutin_days", "days_since_open"}:
        feature_cols.append(extra_feature)
    elif extra_feature in DIFF_FEATURE_BASE_COL:
        df = add_diff_rate_feature(
            df,
            base_col=DIFF_FEATURE_BASE_COL[extra_feature],
            new_col=extra_feature,
            eps=diff_eps,
        )
        feature_cols.append(extra_feature)
    else:
        raise ValueError(f"Unsupported extra_feature: {extra_feature}")
    return df, feature_cols


def fit_feature_stats(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    pred_len: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    min_required = seq_len + pred_len * 3
    train_parts = []
    eligible_wells = []
    for well_id, grp in df.groupby("well_id"):
        grp = grp.sort_values("date")
        if len(grp) < min_required:
            continue
        eligible_wells.append(str(well_id))
        train_end = len(grp) - pred_len * 2
        train_parts.append(grp.iloc[:train_end][feature_cols].values.astype(np.float32))
    if not train_parts:
        raise RuntimeError("No eligible wells for feature fitting.")
    stacked = np.concatenate(train_parts, axis=0)
    means = stacked.mean(axis=0).astype(np.float32)
    stds = stacked.std(axis=0).astype(np.float32)
    stds = np.where(stds > 1e-6, stds, 1.0).astype(np.float32)
    return means, stds, eligible_wells


def build_time_split_data(
    df: pd.DataFrame,
    seq_len: int,
    pred_len: int,
    stride: int,
    feature_cols: List[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[str, np.ndarray, np.ndarray]], List[Tuple[str, np.ndarray, np.ndarray]], List[Dict[str, int]]]:
    train_samples: List[Tuple[np.ndarray, np.ndarray]] = []
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
            train_samples.append((history.astype(np.float32), target.astype(np.float32)))
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

    if not train_samples:
        raise RuntimeError("No train samples were created. Check the dataset length and split settings.")
    if not val_samples or not test_samples:
        raise RuntimeError("Validation or test samples are empty.")
    return train_samples, val_samples, test_samples, well_stats


def evaluate_cases(
    model: nn.Module,
    samples: List[Tuple[str, np.ndarray, np.ndarray]],
    device: torch.device,
) -> Tuple[pd.DataFrame, List[EvalCase], Dict[str, float]]:
    rows = []
    cases: List[EvalCase] = []
    all_true = []
    all_pred = []

    model.eval()
    with torch.no_grad():
        for well_id, history, target in samples:
            x = torch.from_numpy(history).unsqueeze(0).to(device)
            pred = model(x).squeeze(0).cpu().numpy().astype(np.float32)

            sample_mae = float(mean_absolute_error(target, pred))
            sample_rmse = float(math.sqrt(mean_squared_error(target, pred)))
            sample_r2 = float(r2_score(target, pred))

            rows.append(
                {
                    "well_id": well_id,
                    "mae": sample_mae,
                    "rmse": sample_rmse,
                    "r2": sample_r2,
                }
            )
            cases.append(
                EvalCase(
                    well_id=well_id,
                    history=history[:, 0].copy(),
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


def save_case_plot(case: EvalCase, seq_len: int, pred_len: int, save_path: Path, title: str, no_show: bool) -> None:
    x_hist = np.arange(seq_len)
    x_future = np.arange(seq_len, seq_len + pred_len)
    plt.figure(figsize=(12, 5))
    plt.plot(x_hist, case.history, label="History", color="black", alpha=0.7)
    plt.plot(x_future, case.true_future, label="True", color="green", marker=".")
    plt.plot(x_future, case.pred_future, label="Pred", color="red", linestyle="--", marker="x")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Gas Rate (normalized)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    if not no_show:
        plt.show()
    plt.close()


def save_run_config(
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    well_stats: List[Dict[str, int]],
    train_windows: int,
    feature_cols: List[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
) -> None:
    lines = [
        f"timestamp={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"device={device}",
        f"baseline_name=strict_stable_transformer_mlp_head_v2_plus_{args.extra_feature}",
        f"data_dir={args.data_dir}",
        f"extra_feature={args.extra_feature}",
        f"seq_len={args.seq_len}",
        f"pred_len={args.pred_len}",
        f"stride={args.stride}",
        f"batch_size={args.batch_size}",
        f"epochs={args.epochs}",
        f"lr={args.lr}",
        f"weight_decay={args.weight_decay}",
        f"hidden_size={args.hidden_size}",
        f"head_hidden_dim={args.head_hidden_dim}",
        f"num_layers={args.num_layers}",
        f"nhead={args.nhead}",
        f"dropout={args.dropout}",
        f"min_uptime={args.min_uptime}",
        f"diff_eps={args.diff_eps}",
        f"grad_clip_norm={args.grad_clip_norm}",
        f"scheduler=ReduceLROnPlateau",
        f"scheduler_patience={args.scheduler_patience}",
        f"scheduler_factor={args.scheduler_factor}",
        f"scheduler_min_lr={args.scheduler_min_lr}",
        f"early_stopping_patience={args.early_stopping_patience}",
        f"early_stopping_min_delta={args.early_stopping_min_delta}",
        f"min_epochs_before_stop={args.min_epochs_before_stop}",
        f"seed={args.seed}",
        f"eligible_wells={len(well_stats)}",
        f"train_windows={train_windows}",
        f"feature_cols={feature_cols}",
        f"feature_means={feature_means.tolist()}",
        f"feature_stds={feature_stds.tolist()}",
    ]
    (output_dir / "run_config.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(well_stats).sort_values("train_windows", ascending=False).to_csv(
        output_dir / "well_stats.csv", index=False
    )


def save_training_summary(
    output_dir: Path,
    epochs_ran: int,
    best_epoch: int,
    best_val_rmse: float,
    stopped_early: bool,
) -> None:
    lines = [
        f"epochs_ran={epochs_ran}",
        f"best_epoch={best_epoch}",
        f"best_val_mean_rmse={best_val_rmse}",
        f"stopped_early={stopped_early}",
    ]
    (output_dir / "training_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    output_root = Path(args.output_root).resolve() / f"exp_{args.extra_feature}"
    run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    plot_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Extra feature: {args.extra_feature}")

    df, feature_cols = prepare_dataframe(
        data_dir=data_dir,
        min_uptime=args.min_uptime,
        diff_eps=args.diff_eps,
        extra_feature=args.extra_feature,
    )
    feature_means, feature_stds, _ = fit_feature_stats(
        df=df,
        feature_cols=feature_cols,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )
    train_samples, val_samples, test_samples, well_stats = build_time_split_data(
        df=df,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride,
        feature_cols=feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )
    print(f"[INFO] Eligible wells: {len(well_stats)}")
    print(f"[INFO] Train windows: {len(train_samples)}")
    print(f"[INFO] Val wells: {len(val_samples)} | Test wells: {len(test_samples)}")
    save_run_config(args, run_dir, device, well_stats, len(train_samples), feature_cols, feature_means, feature_stds)

    train_loader = DataLoader(
        WindowDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = MultiStepTransformerMLPHead(
        input_dim=len(feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        nhead=args.nhead,
        output_dim=args.pred_len,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )
    criterion = nn.MSELoss()

    best_state = None
    best_epoch = 0
    best_val_rmse = float("inf")
    epochs_without_improvement = 0
    stopped_early = False
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        last_grad_norm = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm))
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        val_df, _, val_summary = evaluate_cases(model, val_samples, device)
        val_rmse = val_summary["mean_rmse"]
        scheduler.step(val_rmse)
        current_lr = float(optimizer.param_groups[0]["lr"])

        improved = val_rmse < (best_val_rmse - args.early_stopping_min_delta)
        if improved:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            val_df.to_csv(run_dir / "best_val_per_well_metrics.csv", index=False)
        else:
            epochs_without_improvement += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_mean_mae": val_summary["mean_mae"],
                "val_mean_rmse": val_summary["mean_rmse"],
                "val_mean_r2": val_summary["mean_r2"],
                "val_pooled_r2": val_summary["pooled_r2"],
                "lr": current_lr,
                "last_grad_norm": last_grad_norm,
                "improved": improved,
                "epochs_without_improvement": epochs_without_improvement,
            }
        )
        print(
            f"[EPOCH {epoch:02d}] "
            f"train_loss={avg_loss:.6f} "
            f"val_mean_mae={val_summary['mean_mae']:.6f} "
            f"val_mean_rmse={val_summary['mean_rmse']:.6f} "
            f"val_mean_r2={val_summary['mean_r2']:.6f} "
            f"lr={current_lr:.6e} "
            f"bad_epochs={epochs_without_improvement}"
        )

        if epoch >= args.min_epochs_before_stop and epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            print(
                f"[INFO] Early stopping triggered at epoch {epoch}. "
                f"Best epoch was {best_epoch} with val_mean_rmse={best_val_rmse:.6f}."
            )
            break

    if best_state is None:
        raise RuntimeError("Training finished without a valid best checkpoint.")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), run_dir / "best_model.pth")
    pd.DataFrame(history_rows).to_csv(run_dir / "training_history.csv", index=False)
    save_training_summary(
        output_dir=run_dir,
        epochs_ran=len(history_rows),
        best_epoch=best_epoch,
        best_val_rmse=best_val_rmse,
        stopped_early=stopped_early,
    )

    val_df, _, val_summary = evaluate_cases(model, val_samples, device)
    test_df, test_cases, test_summary = evaluate_cases(model, test_samples, device)

    val_df.to_csv(run_dir / "val_per_well_metrics.csv", index=False)
    test_df.to_csv(run_dir / "test_per_well_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"split": "val", **val_summary},
            {"split": "test", **test_summary},
        ]
    ).to_csv(run_dir / "summary_metrics.csv", index=False)

    print("\n========== STRICT STABLE SINGLE-FEATURE SUMMARY ==========")
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
    print("======================================================\n")
    print(f"[INFO] Saved run directory: {run_dir}")

    sorted_cases = sorted(test_cases, key=lambda c: c.r2, reverse=True)
    top_k = min(args.plot_top_k, len(sorted_cases))
    for i in range(top_k):
        case = sorted_cases[i]
        save_case_plot(
            case=case,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            save_path=plot_dir / f"best_{i + 1}_{case.well_id}_r2_{case.r2:.4f}.png",
            title=f"Best #{i + 1} | Well {case.well_id} | R2={case.r2:.4f} MAE={case.mae:.4f}",
            no_show=args.no_show,
        )

    worst_case = sorted_cases[-1]
    save_case_plot(
        case=worst_case,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        save_path=plot_dir / f"worst_{worst_case.well_id}_r2_{worst_case.r2:.4f}.png",
        title=f"Worst | Well {worst_case.well_id} | R2={worst_case.r2:.4f} MAE={worst_case.mae:.4f}",
        no_show=args.no_show,
    )


if __name__ == "__main__":
    main()
