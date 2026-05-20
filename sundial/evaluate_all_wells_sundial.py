import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from sundial_finetune_utils import (
    build_eval_samples,
    build_eval_samples_exogenous,
    compute_min_required_points,
    get_well_multivariate_series,
    get_well_series,
    load_clean_well_data,
    load_clean_well_data_with_options,
    mae,
    plot_forecast,
    r2,
    rmse,
    summarize_metrics,
    summarize_metrics_with_pooled,
)


class ExogenousPatchAdapter(nn.Module):
    def __init__(
        self,
        input_token_len: int,
        exo_dim: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_token_len = int(input_token_len)
        self.exo_dim = int(exo_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_token_len * self.exo_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.input_token_len),
        )
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids: torch.Tensor, exo_inputs: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        num_patches = seq_len // self.input_token_len
        exo_patch = exo_inputs.reshape(
            bsz, num_patches, self.input_token_len, self.exo_dim
        ).permute(0, 1, 3, 2)
        exo_patch = exo_patch.reshape(bsz, num_patches, self.input_token_len * self.exo_dim)
        delta_patch = self.mlp(exo_patch)
        delta = delta_patch.reshape(bsz, seq_len)
        return input_ids + torch.tanh(self.gate) * delta


def normalize_exogenous(exo_inputs: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    means = exo_inputs.mean(dim=1, keepdim=True)
    stdev = exo_inputs.std(dim=1, keepdim=True, unbiased=False)
    stdev = torch.where(stdev > eps, stdev, torch.ones_like(stdev))
    return (exo_inputs - means) / stdev


def apply_exogenous_inputs(
    input_ids: torch.Tensor,
    exo_inputs: Optional[torch.Tensor],
    exo_adapter: Optional[ExogenousPatchAdapter],
) -> torch.Tensor:
    if exo_inputs is None or exo_adapter is None:
        return input_ids
    return exo_adapter(input_ids, normalize_exogenous(exo_inputs))


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.dirname(script_dir)
    parser = argparse.ArgumentParser(
        description="Evaluate Sundial on all wells with fixed time split."
    )
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(script_dir, "sundial-base-128m"),
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="",
        help="Optional PEFT adapter path. Empty means baseline model.",
    )
    parser.add_argument("--lookback_len", type=int, default=256)
    parser.add_argument("--forecast_len", type=int, default=96)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--min_uptime", type=float, default=20.0)
    parser.add_argument("--keep_low_uptime", action="store_true")
    parser.add_argument("--include_shutin_days", action="store_true")
    parser.add_argument("--include_prev_shutin_days", action="store_true")
    parser.add_argument("--shutin_uptime_threshold", type=float, default=20.0)
    parser.add_argument("--use_exogenous", action="store_true")
    parser.add_argument("--exo_hidden_dim", type=int, default=128)
    parser.add_argument("--exo_dropout", type=float, default=0.1)
    parser.add_argument(
        "--exo_adapter_path",
        type=str,
        default="",
        help="Optional path to exogenous adapter weights (.pt).",
    )
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--summary_csv", type=str, default="")
    parser.add_argument("--plot_dir", type=str, default="")
    parser.add_argument("--max_plots", type=int, default=3)
    return parser.parse_args()


def load_model(
    model_dir: str,
    adapter_dir: str,
    device: torch.device,
) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    if adapter_dir:
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "adapter_dir is set but `peft` is not installed. Run: pip install peft"
            ) from exc
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.to(device)
    model.eval()
    return model


def build_generate_kwargs(model: torch.nn.Module) -> Dict:
    kwargs = {"do_sample": False, "revin": True}
    eos_id = getattr(model.config, "eos_token_id", None)
    pad_id = getattr(model.config, "pad_token_id", None)
    if eos_id is not None:
        kwargs["pad_token_id"] = eos_id
    elif pad_id is not None:
        kwargs["pad_token_id"] = pad_id
    return kwargs


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    samples,
    device: torch.device,
    forecast_len: int,
    num_samples: int,
    model_name: str,
    plot_dir: str,
    max_plots: int,
    use_exogenous: bool = False,
    exo_adapter: Optional[ExogenousPatchAdapter] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    all_true = []
    all_pred = []
    plot_count = 0
    generate_kwargs = build_generate_kwargs(model)
    if exo_adapter is not None:
        exo_adapter.eval()
    for sample in samples:
        seqs = torch.tensor(sample.context, dtype=torch.float32).unsqueeze(0).to(device)
        if use_exogenous:
            exo = torch.tensor(sample.exo_context, dtype=torch.float32).unsqueeze(0).to(device)
            seqs = apply_exogenous_inputs(seqs, exo, exo_adapter)
        output = model.generate(
            seqs,
            max_new_tokens=forecast_len,
            num_samples=num_samples,
            **generate_kwargs,
        )
        pred_samples = output[0].detach().cpu().numpy()
        pred_mean = np.mean(pred_samples, axis=0)
        pred_low = np.percentile(pred_samples, 5, axis=0)
        pred_high = np.percentile(pred_samples, 95, axis=0)

        row = {
            "model_name": model_name,
            "well_id": sample.well_id,
            "split": sample.split,
            "mae": mae(sample.target, pred_mean),
            "rmse": rmse(sample.target, pred_mean),
            "r2": r2(sample.target, pred_mean),
        }
        rows.append(row)
        all_true.append(sample.target.astype(np.float32))
        all_pred.append(pred_mean.astype(np.float32))

        if sample.split == "test" and plot_dir and plot_count < max_plots:
            os.makedirs(plot_dir, exist_ok=True)
            save_path = os.path.join(plot_dir, f"{model_name}_{sample.well_id}_test.png")
            title = (
                f"{model_name} | Well {sample.well_id} | "
                f"MAE {row['mae']:.4f} | RMSE {row['rmse']:.4f}"
            )
            plot_forecast(
                context=sample.context,
                target=sample.target,
                pred_mean=pred_mean,
                pred_low=pred_low,
                pred_high=pred_high,
                title=title,
                save_path=save_path,
            )
            plot_count += 1

    per_well_df = pd.DataFrame(rows)
    if per_well_df.empty:
        raise RuntimeError("No evaluation rows generated.")
    pooled_true = np.concatenate(all_true, axis=0)
    pooled_pred = np.concatenate(all_pred, axis=0)
    summary = {
        "mean_mae": float(per_well_df["mae"].mean()),
        "mean_rmse": float(per_well_df["rmse"].mean()),
        "mean_r2": float(per_well_df["r2"].mean()),
        "pooled_mae": mae(pooled_true, pooled_pred),
        "pooled_rmse": rmse(pooled_true, pooled_pred),
        "pooled_r2": r2(pooled_true, pooled_pred),
    }
    return per_well_df, summary


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "finetuned" if args.adapter_dir else "baseline"

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model dir: {args.model_dir}")
    if args.adapter_dir:
        print(f"[INFO] Adapter dir: {args.adapter_dir}")

    model = load_model(args.model_dir, args.adapter_dir, device)
    exo_adapter: Optional[ExogenousPatchAdapter] = None
    input_token_len = int(model.config.input_token_len)
    output_token_len = int(model.config.output_token_lens[-1])
    min_points = compute_min_required_points(
        args.lookback_len, output_token_len, args.forecast_len
    )

    df = load_clean_well_data_with_options(
        data_dir=args.data_dir,
        min_uptime=args.min_uptime,
        keep_low_uptime=bool(args.keep_low_uptime),
        add_shutin_days=bool(args.include_shutin_days),
        add_prev_shutin_days=bool(args.include_prev_shutin_days),
        shutin_uptime_threshold=args.shutin_uptime_threshold,
    )
    exogenous_cols: Tuple[str, ...] = ("tubing_pressure", "casing_pressure")
    if args.include_shutin_days:
        exogenous_cols = exogenous_cols + ("shutin_days",)
    if args.include_prev_shutin_days:
        exogenous_cols = exogenous_cols + ("prev_shutin_days",)
    exo_dim = len(exogenous_cols)
    if args.use_exogenous:
        multi_series_dict = get_well_multivariate_series(
            df,
            min_points=min_points,
            exogenous_cols=exogenous_cols,
        )
        series_dict = {k: v["gas_rate"] for k, v in multi_series_dict.items()}
    else:
        series_dict = get_well_series(df, min_points=min_points)
    if not series_dict:
        raise RuntimeError(
            f"No eligible wells found for eval. Need >= {min_points} points per well."
        )

    if args.use_exogenous:
        val_samples = build_eval_samples_exogenous(
            series_dict=multi_series_dict,
            lookback_len=args.lookback_len,
            forecast_len=args.forecast_len,
            split="val",
        )
        test_samples = build_eval_samples_exogenous(
            series_dict=multi_series_dict,
            lookback_len=args.lookback_len,
            forecast_len=args.forecast_len,
            split="test",
        )
        exo_adapter = ExogenousPatchAdapter(
            input_token_len=input_token_len,
            exo_dim=exo_dim,
            hidden_dim=args.exo_hidden_dim,
            dropout=args.exo_dropout,
        ).to(device)
        exo_path = args.exo_adapter_path
        if not exo_path and args.adapter_dir:
            exo_path = os.path.join(args.adapter_dir, "exo_adapter.pt")
        if not exo_path or not os.path.exists(exo_path):
            raise RuntimeError(
                "use_exogenous=True but exo adapter weight file not found. "
                "Provide --exo_adapter_path or ensure adapter_dir/exo_adapter.pt exists."
            )
        exo_adapter.load_state_dict(torch.load(exo_path, map_location=device))
    else:
        val_samples = build_eval_samples(
            series_dict=series_dict,
            lookback_len=args.lookback_len,
            forecast_len=args.forecast_len,
            split="val",
        )
        test_samples = build_eval_samples(
            series_dict=series_dict,
            lookback_len=args.lookback_len,
            forecast_len=args.forecast_len,
            split="test",
        )

    val_df, val_summary = run_eval(
        model=model,
        samples=val_samples,
        device=device,
        forecast_len=args.forecast_len,
        num_samples=args.num_samples,
        model_name=model_name,
        plot_dir="",
        max_plots=0,
        use_exogenous=args.use_exogenous,
        exo_adapter=exo_adapter,
    )
    test_df, test_summary = run_eval(
        model=model,
        samples=test_samples,
        device=device,
        forecast_len=args.forecast_len,
        num_samples=args.num_samples,
        model_name=model_name,
        plot_dir=args.plot_dir,
        max_plots=args.max_plots,
        use_exogenous=args.use_exogenous,
        exo_adapter=exo_adapter,
    )

    per_well_df = pd.concat([val_df, test_df], axis=0, ignore_index=True)
    summary_df = summarize_metrics_with_pooled(
        per_well_df,
        pooled_metrics={
            (model_name, "val"): val_summary,
            (model_name, "test"): test_summary,
        },
    )

    if args.output_csv:
        per_well_df.to_csv(args.output_csv, index=False)
        print(f"[INFO] Saved per-well metrics: {args.output_csv}")
    if args.summary_csv:
        summary_df.to_csv(args.summary_csv, index=False)
        print(f"[INFO] Saved summary metrics: {args.summary_csv}")

    print("\n========== EVAL SUMMARY ==========")
    print(summary_df.to_string(index=False))
    print("==================================\n")


if __name__ == "__main__":
    main()
