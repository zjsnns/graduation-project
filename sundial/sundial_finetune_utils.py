import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RENAME_DICT = {
    "Unnamed: 0": "date",
    "Unnamed: 1": "internal_id",
    "Unnamed: 4": "uptime",
    "气量\n104m3": "gas_rate",
    "油压Mpa": "tubing_pressure",
    "套压Mpa": "casing_pressure",
    "气量\n104m3": "gas_rate",
    "油压Mpa": "tubing_pressure",
    "套压Mpa": "casing_pressure",
    # Mojibake fallbacks from legacy scripts/files
    "姘旈噺\n104m3": "gas_rate",
    "娌瑰帇Mpa": "tubing_pressure",
    "濂楀帇Mpa": "casing_pressure",
}


@dataclass
class EvalSample:
    well_id: str
    split: str
    context: np.ndarray
    target: np.ndarray
    exo_context: Optional[np.ndarray] = None


def load_clean_well_data(data_dir: str, min_uptime: float = 20.0) -> pd.DataFrame:
    return load_clean_well_data_with_options(
        data_dir=data_dir,
        min_uptime=min_uptime,
        keep_low_uptime=False,
        add_shutin_days=False,
        shutin_uptime_threshold=min_uptime,
    )


def _compute_shutin_days(
    dates: pd.Series,
    uptime: pd.Series,
    shutin_uptime_threshold: float,
) -> np.ndarray:
    shutin_days: List[float] = []
    prev_date = None
    prev_streak = 0.0
    for date_value, uptime_value in zip(dates.tolist(), uptime.tolist()):
        is_shutin = bool(pd.notna(uptime_value) and float(uptime_value) <= shutin_uptime_threshold)
        if not is_shutin:
            prev_streak = 0.0
            shutin_days.append(0.0)
        else:
            if prev_date is not None and (pd.Timestamp(date_value) - pd.Timestamp(prev_date)).days == 1:
                prev_streak += 1.0
            else:
                prev_streak = 1.0
            shutin_days.append(prev_streak)
        prev_date = date_value
    return np.asarray(shutin_days, dtype=np.float32)


def _compute_prev_shutin_days(
    dates: pd.Series,
    uptime: pd.Series,
    shutin_uptime_threshold: float,
) -> np.ndarray:
    prev_days: List[float] = []
    current_streak = 0.0
    prev_date = None
    for date_value, uptime_value in zip(dates.tolist(), uptime.tolist()):
        is_shutin = bool(pd.notna(uptime_value) and float(uptime_value) <= shutin_uptime_threshold)
        if is_shutin:
            if prev_date is not None and (pd.Timestamp(date_value) - pd.Timestamp(prev_date)).days == 1:
                current_streak += 1.0
            else:
                current_streak = 1.0
            prev_days.append(0.0)
        else:
            prev_days.append(current_streak)
            current_streak = 0.0
        prev_date = date_value
    return np.asarray(prev_days, dtype=np.float32)


def _compute_days_since_open(dates: pd.Series) -> np.ndarray:
    if len(dates) == 0:
        return np.asarray([], dtype=np.float32)
    first_date = pd.Timestamp(dates.iloc[0])
    return (dates - first_date).dt.days.astype(np.float32).to_numpy()


def load_clean_well_data_with_options(
    data_dir: str,
    min_uptime: float = 20.0,
    keep_low_uptime: bool = False,
    add_shutin_days: bool = False,
    add_prev_shutin_days: bool = False,
    add_days_since_open: bool = False,
    shutin_uptime_threshold: float = 20.0,
) -> pd.DataFrame:
    records: List[pd.DataFrame] = []
    xlsx_files = sorted(
        f for f in os.listdir(data_dir) if f.lower().endswith(".xlsx")
    )

    for file_name in xlsx_files:
        file_path = os.path.join(data_dir, file_name)
        well_id = os.path.splitext(file_name)[0]
        try:
            df = pd.read_excel(file_path, header=1)
        except Exception:
            continue

        df = df.rename(columns=RENAME_DICT)
        required_cols = ["date", "uptime", "gas_rate"]
        if not all(col in df.columns for col in required_cols):
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["uptime"] = pd.to_numeric(df["uptime"], errors="coerce")
        df["gas_rate"] = pd.to_numeric(df["gas_rate"], errors="coerce")
        for col in ["tubing_pressure", "casing_pressure"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = np.nan
        df = df.dropna(subset=["date", "uptime", "gas_rate"]).copy()
        df = df.sort_values("date").reset_index(drop=True)
        if add_days_since_open:
            df["days_since_open"] = _compute_days_since_open(df["date"])
        if add_shutin_days:
            df["shutin_days"] = _compute_shutin_days(
                dates=df["date"],
                uptime=df["uptime"],
                shutin_uptime_threshold=shutin_uptime_threshold,
            )
        if add_prev_shutin_days:
            df["prev_shutin_days"] = _compute_prev_shutin_days(
                dates=df["date"],
                uptime=df["uptime"],
                shutin_uptime_threshold=shutin_uptime_threshold,
            )
        if not keep_low_uptime:
            df = df[df["uptime"] > min_uptime]

        if df.empty:
            continue

        clean_df = pd.DataFrame(
            {
                "date": df["date"],
                "well_id": well_id,
                "gas_rate": df["gas_rate"],
                "tubing_pressure": df["tubing_pressure"],
                "casing_pressure": df["casing_pressure"],
                "uptime": df["uptime"],
            }
        )
        if add_shutin_days:
            clean_df["shutin_days"] = df["shutin_days"].astype(np.float32)
        if add_prev_shutin_days:
            clean_df["prev_shutin_days"] = df["prev_shutin_days"].astype(np.float32)
        if add_days_since_open:
            clean_df["days_since_open"] = df["days_since_open"].astype(np.float32)
        records.append(clean_df)

    if not records:
        raise ValueError(f"No valid .xlsx time series found in: {data_dir}")

    all_df = pd.concat(records, axis=0, ignore_index=True)
    all_df = all_df.sort_values(["well_id", "date"]).reset_index(drop=True)
    return all_df


def compute_min_required_points(
    lookback_len: int, output_token_len: int, forecast_len: int
) -> int:
    return lookback_len + output_token_len + 2 * forecast_len


def get_well_series(
    df: pd.DataFrame,
    min_points: int,
) -> Dict[str, np.ndarray]:
    series_dict: Dict[str, np.ndarray] = {}
    for well_id, grp in df.groupby("well_id"):
        values = grp.sort_values("date")["gas_rate"].values.astype(np.float32)
        if len(values) >= min_points:
            series_dict[well_id] = values
    return series_dict


def get_well_multivariate_series(
    df: pd.DataFrame,
    min_points: int,
    exogenous_cols: Tuple[str, ...] = ("tubing_pressure", "casing_pressure"),
) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for well_id, grp in df.groupby("well_id"):
        grp = grp.sort_values("date").copy()
        gas = grp["gas_rate"].values.astype(np.float32)
        if len(gas) < min_points:
            continue

        exo_arrs: List[np.ndarray] = []
        for col in exogenous_cols:
            if col in grp.columns:
                s = pd.to_numeric(grp[col], errors="coerce")
            else:
                s = pd.Series(np.nan, index=grp.index)
            # Fill missing exogenous values without leaking across wells.
            s = s.ffill().bfill()
            if s.isna().all():
                s = s.fillna(0.0)
            else:
                s = s.fillna(float(s.median()))
            exo_arrs.append(s.values.astype(np.float32))

        exogenous = np.stack(exo_arrs, axis=-1).astype(np.float32)
        out[well_id] = {"gas_rate": gas, "exogenous": exogenous}
    return out


def build_train_windows(
    series_dict: Dict[str, np.ndarray],
    lookback_len: int,
    stride: int,
    forecast_len: int,
    output_token_len: int,
    input_token_len: int,
) -> List[Dict[str, np.ndarray]]:
    windows: List[Dict[str, np.ndarray]] = []
    if lookback_len % input_token_len != 0:
        raise ValueError("lookback_len must be divisible by input_token_len.")

    loss_mask_len = lookback_len // input_token_len
    mask_y = np.zeros(output_token_len, dtype=np.float32)
    mask_y[:forecast_len] = 1.0

    for well_id, series in series_dict.items():
        n = len(series)
        train_end = n - 2 * forecast_len

        max_start = train_end - (lookback_len - input_token_len + output_token_len)
        if max_start < 0:
            continue

        for start in range(0, max_start + 1, stride):
            input_slice = series[start : start + lookback_len]
            label_end = start + lookback_len - input_token_len + output_token_len
            label_slice = series[start:label_end]
            if len(input_slice) != lookback_len or len(label_slice) != (
                lookback_len - input_token_len + output_token_len
            ):
                continue

            windows.append(
                {
                    "well_id": well_id,
                    "input_ids": input_slice.astype(np.float32),
                    "labels": label_slice.astype(np.float32),
                    "loss_masks": np.ones(loss_mask_len, dtype=np.float32),
                    "mask_y": mask_y.copy(),
                }
            )
    return windows


def build_train_windows_exogenous(
    series_dict: Dict[str, Dict[str, np.ndarray]],
    lookback_len: int,
    stride: int,
    forecast_len: int,
    output_token_len: int,
    input_token_len: int,
) -> List[Dict[str, np.ndarray]]:
    windows: List[Dict[str, np.ndarray]] = []
    if lookback_len % input_token_len != 0:
        raise ValueError("lookback_len must be divisible by input_token_len.")

    loss_mask_len = lookback_len // input_token_len
    mask_y = np.zeros(output_token_len, dtype=np.float32)
    mask_y[:forecast_len] = 1.0

    for well_id, bundle in series_dict.items():
        series = bundle["gas_rate"]
        exogenous = bundle["exogenous"]
        n = len(series)
        train_end = n - 2 * forecast_len

        max_start = train_end - (lookback_len - input_token_len + output_token_len)
        if max_start < 0:
            continue

        for start in range(0, max_start + 1, stride):
            input_slice = series[start : start + lookback_len]
            exo_slice = exogenous[start : start + lookback_len]
            label_end = start + lookback_len - input_token_len + output_token_len
            label_slice = series[start:label_end]
            if (
                len(input_slice) != lookback_len
                or len(label_slice) != (lookback_len - input_token_len + output_token_len)
                or exo_slice.shape != (lookback_len, exogenous.shape[-1])
            ):
                continue

            windows.append(
                {
                    "well_id": well_id,
                    "input_ids": input_slice.astype(np.float32),
                    "exo_inputs": exo_slice.astype(np.float32),
                    "labels": label_slice.astype(np.float32),
                    "loss_masks": np.ones(loss_mask_len, dtype=np.float32),
                    "mask_y": mask_y.copy(),
                }
            )
    return windows


def build_eval_samples(
    series_dict: Dict[str, np.ndarray],
    lookback_len: int,
    forecast_len: int,
    split: str,
) -> List[EvalSample]:
    if split not in {"val", "test"}:
        raise ValueError("split must be one of {'val', 'test'}.")

    samples: List[EvalSample] = []
    for well_id, series in series_dict.items():
        n = len(series)
        if split == "val":
            target_start = n - 2 * forecast_len
        else:
            target_start = n - forecast_len
        target_end = target_start + forecast_len
        context_start = target_start - lookback_len
        if context_start < 0:
            continue

        context = series[context_start:target_start]
        target = series[target_start:target_end]
        if len(context) != lookback_len or len(target) != forecast_len:
            continue
        samples.append(
            EvalSample(
                well_id=well_id,
                split=split,
                context=context.astype(np.float32),
                target=target.astype(np.float32),
            )
        )
    return samples


def build_eval_samples_exogenous(
    series_dict: Dict[str, Dict[str, np.ndarray]],
    lookback_len: int,
    forecast_len: int,
    split: str,
) -> List[EvalSample]:
    if split not in {"val", "test"}:
        raise ValueError("split must be one of {'val', 'test'}.")

    samples: List[EvalSample] = []
    for well_id, bundle in series_dict.items():
        series = bundle["gas_rate"]
        exogenous = bundle["exogenous"]
        n = len(series)
        if split == "val":
            target_start = n - 2 * forecast_len
        else:
            target_start = n - forecast_len
        target_end = target_start + forecast_len
        context_start = target_start - lookback_len
        if context_start < 0:
            continue

        context = series[context_start:target_start]
        exo_context = exogenous[context_start:target_start]
        target = series[target_start:target_end]
        if (
            len(context) != lookback_len
            or len(target) != forecast_len
            or exo_context.shape[0] != lookback_len
        ):
            continue
        samples.append(
            EvalSample(
                well_id=well_id,
                split=split,
                context=context.astype(np.float32),
                target=target.astype(np.float32),
                exo_context=exo_context.astype(np.float32),
            )
        )
    return samples


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def summarize_metrics(per_well_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        per_well_df.groupby(["model_name", "split"], as_index=False)[
            ["mae", "rmse", "r2"]
        ]
        .mean(numeric_only=True)
        .sort_values(["model_name", "split"])
    )
    return grouped


def summarize_metrics_with_pooled(
    per_well_df: pd.DataFrame,
    pooled_metrics: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
) -> pd.DataFrame:
    grouped = summarize_metrics(per_well_df)
    grouped = grouped.rename(
        columns={
            "mae": "mean_mae",
            "rmse": "mean_rmse",
            "r2": "mean_r2",
        }
    )
    grouped["pooled_mae"] = np.nan
    grouped["pooled_rmse"] = np.nan
    grouped["pooled_r2"] = np.nan

    if pooled_metrics:
        for idx, row in grouped.iterrows():
            key = (str(row["model_name"]), str(row["split"]))
            extra = pooled_metrics.get(key)
            if not extra:
                continue
            grouped.at[idx, "pooled_mae"] = float(extra["pooled_mae"])
            grouped.at[idx, "pooled_rmse"] = float(extra["pooled_rmse"])
            grouped.at[idx, "pooled_r2"] = float(extra["pooled_r2"])
    return grouped


def timestamp_run_dir(base_output_dir: str) -> str:
    os.makedirs(base_output_dir, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("run_%Y%m%d_%H%M")
    run_dir = os.path.join(base_output_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "adapter"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    return run_dir


def save_config_yaml(path: str, cfg: Dict) -> None:
    # JSON is valid YAML 1.2.
    import json

    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg, indent=2, ensure_ascii=False))


def plot_forecast(
    context: np.ndarray,
    target: np.ndarray,
    pred_mean: np.ndarray,
    pred_low: np.ndarray,
    pred_high: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    hist_display_len = min(200, len(context))
    x_hist = np.arange(-hist_display_len, 0)
    x_pred = np.arange(0, len(target))

    plt.figure(figsize=(12, 6))
    plt.plot(x_hist, context[-hist_display_len:], label="History", color="blue")
    plt.plot(x_pred, target, label="Ground Truth", color="green", linestyle="--")
    plt.plot(x_pred, pred_mean, label="Prediction Mean", color="red")
    plt.fill_between(
        x_pred,
        pred_low,
        pred_high,
        color="red",
        alpha=0.2,
        label="90% Prediction Interval",
    )
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Gas Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
