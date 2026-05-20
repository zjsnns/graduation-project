import argparse
import math
import os
import random
import shutil
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

from sundial_finetune_utils import (
    build_eval_samples,
    build_eval_samples_exogenous,
    build_train_windows,
    build_train_windows_exogenous,
    compute_min_required_points,
    get_well_multivariate_series,
    get_well_series,
    load_clean_well_data,
    load_clean_well_data_with_options,
    mae,
    plot_forecast,
    r2,
    rmse,
    save_config_yaml,
    summarize_metrics,
    summarize_metrics_with_pooled,
    timestamp_run_dir,
)


class SundialTrainDataset(Dataset):
    def __init__(self, windows: List[Dict[str, np.ndarray]]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self.windows[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    keys = ["input_ids", "labels", "loss_masks", "mask_y"]
    if "exo_inputs" in batch[0]:
        keys.append("exo_inputs")
    out = {}
    for key in keys:
        out[key] = torch.tensor(
            np.stack([sample[key] for sample in batch], axis=0),
            dtype=torch.float32,
        )
    return out


class ExogenousPatchAdapter(nn.Module):
    def __init__(
        self,
        input_token_len: int,
        exo_dim: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        gate_init: float = 0.0,
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
        # Start from zero exogenous effect; let training learn how much to use.
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, input_ids: torch.Tensor, exo_inputs: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [B, T].")
        if exo_inputs.dim() != 3:
            raise ValueError("exo_inputs must have shape [B, T, C].")

        bsz, seq_len = input_ids.shape
        if exo_inputs.shape[0] != bsz or exo_inputs.shape[1] != seq_len:
            raise ValueError("exo_inputs and input_ids batch/time dimensions must match.")
        if exo_inputs.shape[2] != self.exo_dim:
            raise ValueError(f"Expected exo dim {self.exo_dim}, got {exo_inputs.shape[2]}.")
        if seq_len % self.input_token_len != 0:
            raise ValueError("Sequence length must be divisible by input_token_len.")

        num_patches = seq_len // self.input_token_len
        exo_patch = exo_inputs.reshape(
            bsz, num_patches, self.input_token_len, self.exo_dim
        )
        exo_patch = exo_patch.permute(0, 1, 3, 2).reshape(
            bsz, num_patches, self.input_token_len * self.exo_dim
        )
        delta_patch = self.mlp(exo_patch)
        delta = delta_patch.reshape(bsz, seq_len)
        return input_ids + torch.tanh(self.gate) * delta


def normalize_exogenous(exo_inputs: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    means = exo_inputs.mean(dim=1, keepdim=True).detach()
    stdev = exo_inputs.std(dim=1, keepdim=True, unbiased=False).detach()
    stdev = torch.where(stdev > eps, stdev, torch.ones_like(stdev))
    return (exo_inputs - means) / stdev


def apply_exogenous_inputs(
    input_ids: torch.Tensor,
    exo_inputs: Optional[torch.Tensor],
    exo_adapter: Optional[ExogenousPatchAdapter],
) -> torch.Tensor:
    if exo_inputs is None or exo_adapter is None:
        return input_ids
    exo_norm = normalize_exogenous(exo_inputs)
    return exo_adapter(input_ids, exo_norm)


def build_well_balanced_sampler(
    windows: List[Dict[str, np.ndarray]], power: float, seed: int
) -> Tuple[WeightedRandomSampler, pd.Series]:
    well_ids = [str(w["well_id"]) for w in windows]
    counts = pd.Series(well_ids).value_counts()
    if counts.empty:
        raise RuntimeError("Cannot build sampler: no training windows.")

    weights = np.array([(1.0 / counts[wid]) ** power for wid in well_ids], dtype=np.float64)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(windows),
        replacement=True,
        generator=generator,
    )
    return sampler, counts


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Sundial.")
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(script_dir, "sundial-base-128m"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(script_dir, "outputs"),
    )
    parser.add_argument("--lookback_len", type=int, default=256)
    parser.add_argument("--forecast_len", type=int, default=96)
    parser.add_argument(
        "--output_head_len",
        type=int,
        default=0,
        help="0 keeps pretrained head length (720). Set 96 to train a 96-step head.",
    )
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--min_uptime", type=float, default=20.0)
    parser.add_argument(
        "--keep_low_uptime",
        action="store_true",
        help="Keep rows with uptime <= min_uptime instead of dropping them.",
    )
    parser.add_argument(
        "--include_shutin_days",
        action="store_true",
        help="Add consecutive shut-in days as an exogenous feature.",
    )
    parser.add_argument(
        "--include_prev_shutin_days",
        action="store_true",
        help="Add the number of consecutive shut-in days before the current production day as an exogenous feature.",
    )
    parser.add_argument(
        "--shutin_uptime_threshold",
        type=float,
        default=20.0,
        help="Rows with uptime <= threshold are counted as shut-in days.",
    )

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--eval_max_plots", type=int, default=3)
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--experiment_log", type=str, default="")
    parser.add_argument(
        "--full_finetune",
        action="store_true",
        help="Train all base-model parameters instead of LoRA adapters.",
    )
    parser.add_argument(
        "--use_exogenous",
        action="store_true",
        help="Use tubing/casing pressure as exogenous inputs via a trainable adapter.",
    )
    parser.add_argument("--exo_hidden_dim", type=int, default=128)
    parser.add_argument("--exo_dropout", type=float, default=0.1)
    parser.add_argument("--exo_gate_init", type=float, default=0.0)
    parser.add_argument(
        "--exo_lr_mult",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for exogenous adapter params.",
    )
    parser.add_argument(
        "--well_balanced_sampling",
        action="store_true",
        help="Use inverse-frequency sampling over well_id to avoid large-well dominance.",
    )
    parser.add_argument(
        "--well_balance_power",
        type=float,
        default=1.0,
        help="Sampling weight exponent. weight=(1/n_windows_per_well)^power",
    )
    parser.add_argument(
        "--train_flow_loss_head",
        action="store_true",
        help="Unfreeze flow_loss parameters. Auto-enabled when output head is overridden.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help="0 means no cap.",
    )
    parser.set_defaults(fp16=True, gradient_checkpointing=True)
    return parser.parse_args()


def require_peft():
    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Missing dependency `peft`. Install with: pip install peft"
        ) from exc
    return LoraConfig, PeftModel, TaskType, get_peft_model


def append_log(log_path: str, text: str) -> None:
    if not log_path:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def check_data_invariants(
    df: pd.DataFrame,
    windows: List[Dict[str, np.ndarray]],
    lookback_len: int,
    output_token_len: int,
    forecast_len: int,
    input_token_len: int,
    use_exogenous: bool = False,
    exo_dim: int = 2,
) -> None:
    for _, grp in df.groupby("well_id"):
        if not grp["date"].is_monotonic_increasing:
            raise AssertionError("Date order is not monotonic for at least one well.")

    if lookback_len % input_token_len != 0:
        raise AssertionError("lookback_len must be divisible by input_token_len.")

    expected_label_len = lookback_len - input_token_len + output_token_len
    expected_loss_mask_len = lookback_len // input_token_len

    for i, sample in enumerate(windows[:20]):
        if sample["input_ids"].shape[0] != lookback_len:
            raise AssertionError(f"Window {i} input_ids shape mismatch.")
        if sample["labels"].shape[0] != expected_label_len:
            raise AssertionError(f"Window {i} labels shape mismatch.")
        if sample["loss_masks"].shape[0] != expected_loss_mask_len:
            raise AssertionError(f"Window {i} loss_masks shape mismatch.")
        if sample["mask_y"].shape[0] != output_token_len:
            raise AssertionError(f"Window {i} mask_y shape mismatch.")
        if not np.all(sample["mask_y"][:forecast_len] == 1.0):
            raise AssertionError(f"Window {i} mask_y first forecast part mismatch.")
        if not np.all(sample["mask_y"][forecast_len:] == 0.0):
            raise AssertionError(f"Window {i} mask_y tail mismatch.")
        if use_exogenous:
            if "exo_inputs" not in sample:
                raise AssertionError(f"Window {i} missing exo_inputs.")
            if sample["exo_inputs"].shape != (lookback_len, exo_dim):
                raise AssertionError(f"Window {i} exo_inputs shape mismatch.")


def move_batch_to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def build_generate_kwargs(model: torch.nn.Module) -> Dict:
    kwargs = {"do_sample": False, "revin": True}
    eos_id = getattr(model.config, "eos_token_id", None)
    pad_id = getattr(model.config, "pad_token_id", None)
    if eos_id is not None:
        kwargs["pad_token_id"] = eos_id
    elif pad_id is not None:
        kwargs["pad_token_id"] = pad_id
    return kwargs


def maybe_override_output_head(
    model: torch.nn.Module, output_head_len: int
) -> Tuple[int, bool]:
    current_len = int(model.config.output_token_lens[-1])
    if output_head_len <= 0 or output_head_len == current_len:
        return current_len, False

    if not hasattr(model, "flow_loss"):
        raise AttributeError("Model has no flow_loss; cannot override output head.")

    flow_loss_cls = model.flow_loss.__class__
    hidden_size = int(model.config.hidden_size)
    flow_depth = int(model.config.flow_loss_depth)
    sampling_steps = int(model.config.num_sampling_steps)
    new_flow_loss = flow_loss_cls(
        output_head_len,
        hidden_size,
        flow_depth,
        hidden_size,
        sampling_steps,
    ).to(next(model.parameters()).device)

    model.flow_loss = new_flow_loss
    model.config.output_token_lens = [int(output_head_len)]
    return int(output_head_len), True


def get_flow_loss_module(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "flow_loss"):
        return model.flow_loss
    if hasattr(model, "get_base_model"):
        base = model.get_base_model()
        if hasattr(base, "flow_loss"):
            return base.flow_loss
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "model") and hasattr(base.model, "flow_loss"):
            return base.model.flow_loss
        if hasattr(base, "flow_loss"):
            return base.flow_loss
    raise AttributeError("Cannot locate flow_loss module on model.")


@torch.no_grad()
def evaluate_samples(
    model: torch.nn.Module,
    samples,
    device: torch.device,
    forecast_len: int,
    num_samples: int,
    model_name: str,
    plot_dir: str = "",
    max_plots: int = 0,
    use_exogenous: bool = False,
    exo_adapter: Optional[ExogenousPatchAdapter] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    model.eval()
    if exo_adapter is not None:
        exo_adapter.eval()
    rows = []
    all_true = []
    all_pred = []
    plot_count = 0
    generate_kwargs = build_generate_kwargs(model)

    for sample in samples:
        seqs = torch.tensor(sample.context, dtype=torch.float32).unsqueeze(0).to(device)
        if use_exogenous:
            if sample.exo_context is None:
                raise RuntimeError("use_exogenous=True but sample has no exo_context.")
            exo = (
                torch.tensor(sample.exo_context, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
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
            plot_path = os.path.join(plot_dir, f"{model_name}_{sample.well_id}_test.png")
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
                save_path=plot_path,
            )
            plot_count += 1

    per_well_df = pd.DataFrame(rows)
    if per_well_df.empty:
        raise RuntimeError("No evaluation rows generated.")
    avg_mae = float(per_well_df["mae"].mean())
    avg_rmse = float(per_well_df["rmse"].mean())
    avg_r2 = float(per_well_df["r2"].mean())
    pooled_true = np.concatenate(all_true, axis=0)
    pooled_pred = np.concatenate(all_pred, axis=0)
    return per_well_df, {
        "mae": avg_mae,
        "rmse": avg_rmse,
        "r2": avg_r2,
        "pooled_mae": mae(pooled_true, pooled_pred),
        "pooled_rmse": rmse(pooled_true, pooled_pred),
        "pooled_r2": r2(pooled_true, pooled_pred),
    }


def train_one_epoch(
    model: torch.nn.Module,
    exo_adapter: Optional[ExogenousPatchAdapter],
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_accum: int,
    max_grad_norm: float,
    use_fp16: bool,
    max_train_steps: int,
    use_exogenous: bool = False,
) -> Tuple[float, int]:
    model.train()
    if exo_adapter is not None:
        exo_adapter.train()
    losses = []
    optimizer.zero_grad(set_to_none=True)
    update_steps = 0

    for step, batch in enumerate(train_loader):
        batch = move_batch_to_device(batch, device)
        use_amp = use_fp16 and device.type == "cuda"
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
            if use_amp
            else nullcontext()
        )
        with amp_ctx:
            input_ids = batch["input_ids"]
            if use_exogenous:
                if "exo_inputs" not in batch:
                    raise RuntimeError("use_exogenous=True but batch has no exo_inputs.")
                input_ids = apply_exogenous_inputs(
                    input_ids, batch["exo_inputs"], exo_adapter
                )
            outputs = model(
                input_ids=input_ids,
                labels=batch["labels"],
                loss_masks=batch["loss_masks"],
                mask_y=batch["mask_y"],
                revin=True,
            )
            loss = outputs.loss / grad_accum

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_steps += 1
            if max_train_steps > 0 and update_steps >= max_train_steps:
                losses.append(float(loss.item() * grad_accum))
                break

        losses.append(float(loss.item() * grad_accum))

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return mean_loss, update_steps


def load_base_model(
    model_dir: str,
    device: torch.device,
    output_head_len: int = 0,
) -> Tuple[torch.nn.Module, int, bool]:
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    head_len, resized = maybe_override_output_head(model, output_head_len)
    return model, head_len, resized


def copy_remote_code_files(model_dir: str, checkpoint_dir: str) -> None:
    for name in [
        "modeling_sundial.py",
        "configuration_sundial.py",
        "flow_loss.py",
        "ts_generation_mixin.py",
    ]:
        src = os.path.join(model_dir, name)
        dst = os.path.join(checkpoint_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


def load_full_checkpoint_weights(
    model: torch.nn.Module, checkpoint_dir: str, device: torch.device
) -> None:
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file

        state_dict = load_file(safetensors_path, device=str(device))
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location=device)
    else:
        raise FileNotFoundError(
            f"No model weights found in checkpoint dir: {checkpoint_dir}"
        )
    model.load_state_dict(state_dict, strict=True)


def main() -> None:
    args = parse_args()
    if args.no_fp16:
        args.fp16 = False
    if args.no_gradient_checkpointing:
        args.gradient_checkpointing = False

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = timestamp_run_dir(args.output_dir)
    plot_dir = os.path.join(run_dir, "plots")
    adapter_dir = os.path.join(run_dir, "adapter")
    full_model_dir = os.path.join(run_dir, "full_model")
    best_adapter_dir = (
        os.path.join(full_model_dir, "best")
        if args.full_finetune
        else os.path.join(adapter_dir, "best")
    )
    best_flow_loss_path = os.path.join(best_adapter_dir, "flow_loss_head.pt")
    best_exo_adapter_path = os.path.join(best_adapter_dir, "exo_adapter.pt")
    os.makedirs(best_adapter_dir, exist_ok=True)
    if not args.experiment_name:
        args.experiment_name = os.path.basename(run_dir)
    if not args.experiment_log:
        args.experiment_log = os.path.join(args.output_dir, "finetune_experiments.txt")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Run dir: {run_dir}")
    append_log(
        args.experiment_log,
        (
            f"\n=== {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | START | "
            f"{args.experiment_name} ===\n"
            f"run_dir={run_dir}\n"
            f"params: lookback={args.lookback_len}, forecast={args.forecast_len}, "
            f"stride={args.stride}, epochs={args.epochs}, lr={args.lr}, "
            f"batch={args.batch_size}, grad_accum={args.grad_accum}, "
            f"full_finetune={args.full_finetune}, "
            f"lora_r={args.lora_r}, lora_alpha={args.lora_alpha}, "
            f"lora_dropout={args.lora_dropout}, fp16={args.fp16}, seed={args.seed}, "
            f"output_head_len={args.output_head_len}, "
            f"use_exogenous={args.use_exogenous}, "
            f"min_uptime={args.min_uptime}, keep_low_uptime={args.keep_low_uptime}, "
            f"include_shutin_days={args.include_shutin_days}, "
            f"include_prev_shutin_days={args.include_prev_shutin_days}, "
            f"shutin_uptime_threshold={args.shutin_uptime_threshold}, "
            f"exo_hidden_dim={args.exo_hidden_dim}, "
            f"exo_dropout={args.exo_dropout}, exo_gate_init={args.exo_gate_init}, "
            f"exo_lr_mult={args.exo_lr_mult}, "
            f"well_balanced_sampling={args.well_balanced_sampling}, "
            f"well_balance_power={args.well_balance_power}"
        ),
    )

    # 1) Data load & checks
    df = load_clean_well_data_with_options(
        data_dir=args.data_dir,
        min_uptime=args.min_uptime,
        keep_low_uptime=bool(args.keep_low_uptime),
        add_shutin_days=bool(args.include_shutin_days),
        add_prev_shutin_days=bool(args.include_prev_shutin_days),
        shutin_uptime_threshold=args.shutin_uptime_threshold,
    )
    # Baseline always keeps the native pretrained head for fair comparison.
    baseline_model, baseline_head_len, _ = load_base_model(
        args.model_dir, device, output_head_len=0
    )
    # Training model may override output head length (e.g., 96).
    base_model_for_shape, output_token_len, resized_head = load_base_model(
        args.model_dir, device, output_head_len=args.output_head_len
    )
    if resized_head:
        print(
            f"[INFO] Output head overridden: pretrained={baseline_head_len}, "
            f"train_head={output_token_len}"
        )
    input_token_len = int(base_model_for_shape.config.input_token_len)
    min_points = compute_min_required_points(
        args.lookback_len, output_token_len, args.forecast_len
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
            f"No eligible wells found. Need at least {min_points} points per well."
        )

    if args.use_exogenous:
        train_windows = build_train_windows_exogenous(
            series_dict=multi_series_dict,
            lookback_len=args.lookback_len,
            stride=args.stride,
            forecast_len=args.forecast_len,
            output_token_len=output_token_len,
            input_token_len=input_token_len,
        )
    else:
        train_windows = build_train_windows(
            series_dict=series_dict,
            lookback_len=args.lookback_len,
            stride=args.stride,
            forecast_len=args.forecast_len,
            output_token_len=output_token_len,
            input_token_len=input_token_len,
        )
    if not train_windows:
        raise RuntimeError("No training windows built. Check lookback/stride/data quality.")

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
    if not val_samples or not test_samples:
        raise RuntimeError("Validation/Test samples are empty after split.")

    check_data_invariants(
        df=df,
        windows=train_windows,
        lookback_len=args.lookback_len,
        output_token_len=output_token_len,
        forecast_len=args.forecast_len,
        input_token_len=input_token_len,
        use_exogenous=args.use_exogenous,
        exo_dim=exo_dim,
    )

    eligible_df = pd.DataFrame(
        [{"well_id": w, "points": int(len(s))} for w, s in series_dict.items()]
    ).sort_values("points", ascending=False)
    eligible_df.to_csv(os.path.join(run_dir, "eligible_wells.csv"), index=False)
    train_window_counts = pd.Series(
        [str(w["well_id"]) for w in train_windows], name="well_id"
    ).value_counts()
    train_window_counts.rename("window_count").to_csv(
        os.path.join(run_dir, "train_window_counts.csv"), index_label="well_id"
    )

    print(
        f"[INFO] Eligible wells: {len(series_dict)} | "
        f"train windows: {len(train_windows)} | "
        f"val samples: {len(val_samples)} | test samples: {len(test_samples)}"
    )
    append_log(
        args.experiment_log,
        (
            f"data: eligible_wells={len(series_dict)}, train_windows={len(train_windows)}, "
            f"min_windows_per_well={int(train_window_counts.min())}, "
            f"max_windows_per_well={int(train_window_counts.max())}, "
            f"use_exogenous={args.use_exogenous}"
        ),
    )

    # 2) Baseline eval with base model
    baseline_val_df, baseline_val_avg = evaluate_samples(
        model=baseline_model,
        samples=val_samples,
        device=device,
        forecast_len=args.forecast_len,
        num_samples=args.num_samples,
        model_name="baseline",
        use_exogenous=False,
    )
    baseline_test_df, baseline_test_avg = evaluate_samples(
        model=baseline_model,
        samples=test_samples,
        device=device,
        forecast_len=args.forecast_len,
        num_samples=args.num_samples,
        model_name="baseline",
        plot_dir=plot_dir,
        max_plots=args.eval_max_plots,
        use_exogenous=False,
    )
    print(
        f"[BASELINE] val_mae={baseline_val_avg['mae']:.4f} "
        f"test_mae={baseline_test_avg['mae']:.4f}"
    )
    append_log(
        args.experiment_log,
        (
            f"baseline: val_mae={baseline_val_avg['mae']:.6f}, "
            f"val_rmse={baseline_val_avg['rmse']:.6f}, val_r2={baseline_val_avg['r2']:.6f}, "
            f"val_pooled_r2={baseline_val_avg['pooled_r2']:.6f}, "
            f"test_mae={baseline_test_avg['mae']:.6f}, "
            f"test_rmse={baseline_test_avg['rmse']:.6f}, test_r2={baseline_test_avg['r2']:.6f}, "
            f"test_pooled_r2={baseline_test_avg['pooled_r2']:.6f}"
        ),
    )
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 3) Fine-tuning setup (LoRA or full)
    PeftModel = None
    if args.full_finetune:
        model = base_model_for_shape
        for p in model.parameters():
            p.requires_grad = True
        print("[INFO] Full fine-tuning mode enabled: training all model parameters.")
    else:
        LoraConfig, PeftModel, TaskType, get_peft_model = require_peft()
        lora_targets = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=lora_targets,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model_for_shape, lora_config)
    exo_adapter: Optional[ExogenousPatchAdapter] = None
    if args.use_exogenous:
        exo_adapter = ExogenousPatchAdapter(
            input_token_len=input_token_len,
            exo_dim=exo_dim,
            hidden_dim=args.exo_hidden_dim,
            dropout=args.exo_dropout,
            gate_init=args.exo_gate_init,
        ).to(device)
        exo_trainable = sum(p.numel() for p in exo_adapter.parameters() if p.requires_grad)
        print(f"[INFO] Exogenous adapter enabled: trainable params={exo_trainable}")
    train_flow_loss_active = bool(resized_head or args.train_flow_loss_head)
    if train_flow_loss_active:
        trainable_flow_params = 0
        for name, p in model.named_parameters():
            if "flow_loss" in name:
                p.requires_grad = True
                trainable_flow_params += p.numel()
        print(
            f"[INFO] flow_loss parameters unfrozen: {trainable_flow_params} params "
            f"(resized_head={resized_head})"
        )
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            # Required by PEFT + gradient checkpointing to keep graph connected.
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        except Exception as exc:
            print(
                f"[WARN] Gradient checkpointing disabled due to incompatibility: {exc}"
            )
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            args.gradient_checkpointing = False
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"trainable params: {trainable:,} || all params: {total:,} || "
            f"trainable%: {100.0 * trainable / max(total, 1):.4f}"
        )

    train_dataset = SundialTrainDataset(train_windows)
    if args.well_balanced_sampling:
        sampler, window_counts = build_well_balanced_sampler(
            train_windows,
            power=max(float(args.well_balance_power), 0.0),
            seed=args.seed,
        )
        print(
            f"[INFO] Well-balanced sampling enabled | wells={len(window_counts)} "
            f"window_count[min/median/max]={int(window_counts.min())}/"
            f"{float(window_counts.median()):.1f}/{int(window_counts.max())}"
        )
        append_log(
            args.experiment_log,
            (
                f"sampler: well_balanced=True, power={args.well_balance_power}, "
                f"wells={len(window_counts)}, min_win={int(window_counts.min())}, "
                f"max_win={int(window_counts.max())}"
            ),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=0,
            collate_fn=collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )

    train_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_train_steps = train_steps_per_epoch * args.epochs
    warmup_steps = int(total_train_steps * args.warmup_ratio)

    optimizer_groups = [
        {
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
    ]
    if exo_adapter is not None:
        optimizer_groups.append(
            {
                "params": [p for p in exo_adapter.parameters() if p.requires_grad],
                "lr": float(args.lr) * float(args.exo_lr_mult),
                "weight_decay": args.weight_decay,
            }
        )
    optimizer = torch.optim.AdamW(optimizer_groups)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_train_steps, 1),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    history = []
    best_val_mae = float("inf")

    # 4) Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_updates = train_one_epoch(
            model=model,
            exo_adapter=exo_adapter,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            grad_accum=args.grad_accum,
            max_grad_norm=args.max_grad_norm,
            use_fp16=args.fp16,
            max_train_steps=args.max_train_steps,
            use_exogenous=args.use_exogenous,
        )
        val_df, val_avg = evaluate_samples(
            model=model,
            samples=val_samples,
            device=device,
            forecast_len=args.forecast_len,
            num_samples=args.num_samples,
            model_name="finetuned",
            use_exogenous=args.use_exogenous,
            exo_adapter=exo_adapter,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mae": val_avg["mae"],
                "val_rmse": val_avg["rmse"],
                "val_r2": val_avg["r2"],
                "val_pooled_r2": val_avg["pooled_r2"],
                "updates": train_updates,
            }
        )
        print(
            f"[EPOCH {epoch}] loss={train_loss:.6f} "
            f"val_mae={val_avg['mae']:.4f} val_rmse={val_avg['rmse']:.4f}"
        )
        append_log(
            args.experiment_log,
            (
                f"epoch={epoch}: train_loss={train_loss:.6f}, "
                f"val_mae={val_avg['mae']:.6f}, val_rmse={val_avg['rmse']:.6f}, "
                f"val_r2={val_avg['r2']:.6f}, val_pooled_r2={val_avg['pooled_r2']:.6f}, "
                f"updates={train_updates}"
            ),
        )

        if val_avg["mae"] < best_val_mae:
            best_val_mae = val_avg["mae"]
            model.save_pretrained(best_adapter_dir)
            if args.full_finetune:
                copy_remote_code_files(args.model_dir, best_adapter_dir)
            if train_flow_loss_active:
                flow_module = get_flow_loss_module(model)
                torch.save(flow_module.state_dict(), best_flow_loss_path)
            if exo_adapter is not None:
                torch.save(exo_adapter.state_dict(), best_exo_adapter_path)
            if args.full_finetune:
                print(f"[INFO] Saved best full model to: {best_adapter_dir}")
            else:
                print(f"[INFO] Saved best adapter to: {best_adapter_dir}")

        if args.max_train_steps > 0 and train_updates >= args.max_train_steps:
            print("[INFO] max_train_steps reached; ending early.")
            break

    pd.DataFrame(history).to_csv(os.path.join(run_dir, "train_history.csv"), index=False)

    # 5) Reload best checkpoint and evaluate finetuned
    del model
    del exo_adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.full_finetune:
        best_model, _, _ = load_base_model(
            args.model_dir, device, output_head_len=args.output_head_len
        )
        load_full_checkpoint_weights(best_model, best_adapter_dir, device)
    else:
        base_model, _, _ = load_base_model(
            args.model_dir, device, output_head_len=args.output_head_len
        )
        if os.path.exists(best_flow_loss_path):
            flow_module = get_flow_loss_module(base_model)
            flow_module.load_state_dict(
                torch.load(best_flow_loss_path, map_location=device)
            )
        if PeftModel is None:
            raise RuntimeError("PEFT is required to load LoRA adapters.")
        best_model = PeftModel.from_pretrained(base_model, best_adapter_dir)
        best_model.to(device)
    best_exo_adapter: Optional[ExogenousPatchAdapter] = None
    if args.use_exogenous:
        best_exo_adapter = ExogenousPatchAdapter(
            input_token_len=input_token_len,
            exo_dim=exo_dim,
            hidden_dim=args.exo_hidden_dim,
            dropout=args.exo_dropout,
            gate_init=args.exo_gate_init,
        ).to(device)
        if os.path.exists(best_exo_adapter_path):
            best_exo_adapter.load_state_dict(
                torch.load(best_exo_adapter_path, map_location=device)
            )
        best_exo_adapter.eval()

    finetuned_val_df, finetuned_val_avg = evaluate_samples(
        model=best_model,
        samples=val_samples,
        device=device,
        forecast_len=args.forecast_len,
        num_samples=args.num_samples,
        model_name="finetuned",
        use_exogenous=args.use_exogenous,
        exo_adapter=best_exo_adapter,
    )
    finetuned_test_df, finetuned_test_avg = evaluate_samples(
        model=best_model,
        samples=test_samples,
        device=device,
        forecast_len=args.forecast_len,
        num_samples=args.num_samples,
        model_name="finetuned",
        plot_dir=plot_dir,
        max_plots=args.eval_max_plots,
        use_exogenous=args.use_exogenous,
        exo_adapter=best_exo_adapter,
    )

    # 6) Save report artifacts
    per_well_df = pd.concat(
        [
            baseline_val_df,
            baseline_test_df,
            finetuned_val_df,
            finetuned_test_df,
        ],
        axis=0,
        ignore_index=True,
    )
    per_well_df.to_csv(os.path.join(run_dir, "per_well_metrics.csv"), index=False)

    summary_df = summarize_metrics_with_pooled(
        per_well_df,
        pooled_metrics={
            ("baseline", "val"): baseline_val_avg,
            ("baseline", "test"): baseline_test_avg,
            ("finetuned", "val"): finetuned_val_avg,
            ("finetuned", "test"): finetuned_test_avg,
        },
    )
    summary_df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    cfg = {
        "data_dir": args.data_dir,
        "model_dir": args.model_dir,
        "output_dir": args.output_dir,
        "lookback_len": args.lookback_len,
        "forecast_len": args.forecast_len,
        "stride": args.stride,
        "epochs": args.epochs,
        "lr": args.lr,
        "full_finetune": bool(args.full_finetune),
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "min_uptime": args.min_uptime,
        "keep_low_uptime": bool(args.keep_low_uptime),
        "include_shutin_days": bool(args.include_shutin_days),
        "include_prev_shutin_days": bool(args.include_prev_shutin_days),
        "shutin_uptime_threshold": float(args.shutin_uptime_threshold),
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "output_head_len_arg": args.output_head_len,
        "head_overridden": resized_head,
        "train_flow_loss_head": train_flow_loss_active,
        "baseline_head_len": baseline_head_len,
        "input_token_len": input_token_len,
        "output_token_len": output_token_len,
        "eligible_wells": int(len(series_dict)),
        "train_windows": int(len(train_windows)),
        "min_points_per_well": int(min_points),
        "well_balanced_sampling": bool(args.well_balanced_sampling),
        "well_balance_power": float(args.well_balance_power),
        "use_exogenous": bool(args.use_exogenous),
        "exogenous_cols": list(exogenous_cols),
        "exogenous_dim": int(exo_dim),
        "exo_hidden_dim": int(args.exo_hidden_dim),
        "exo_dropout": float(args.exo_dropout),
        "exo_gate_init": float(args.exo_gate_init),
        "exo_lr_mult": float(args.exo_lr_mult),
        "run_dir": run_dir,
    }
    save_config_yaml(os.path.join(run_dir, "config.yaml"), cfg)

    print("\n========== SUMMARY ==========")
    print(
        f"Baseline test  | MAE {baseline_test_avg['mae']:.4f} "
        f"RMSE {baseline_test_avg['rmse']:.4f} R2 {baseline_test_avg['r2']:.4f} "
        f"PooledR2 {baseline_test_avg['pooled_r2']:.4f}"
    )
    print(
        f"Finetuned test | MAE {finetuned_test_avg['mae']:.4f} "
        f"RMSE {finetuned_test_avg['rmse']:.4f} R2 {finetuned_test_avg['r2']:.4f} "
        f"PooledR2 {finetuned_test_avg['pooled_r2']:.4f}"
    )
    print(f"Artifacts saved to: {run_dir}")
    print("============================\n")
    append_log(
        args.experiment_log,
        (
            f"final: finetuned_val_mae={finetuned_val_avg['mae']:.6f}, "
            f"finetuned_val_rmse={finetuned_val_avg['rmse']:.6f}, "
            f"finetuned_val_r2={finetuned_val_avg['r2']:.6f}, "
            f"finetuned_val_pooled_r2={finetuned_val_avg['pooled_r2']:.6f}, "
            f"finetuned_test_mae={finetuned_test_avg['mae']:.6f}, "
            f"finetuned_test_rmse={finetuned_test_avg['rmse']:.6f}, "
            f"finetuned_test_r2={finetuned_test_avg['r2']:.6f}, "
            f"finetuned_test_pooled_r2={finetuned_test_avg['pooled_r2']:.6f}, "
            f"delta_test_mae={finetuned_test_avg['mae'] - baseline_test_avg['mae']:.6f}, "
            f"delta_test_rmse={finetuned_test_avg['rmse'] - baseline_test_avg['rmse']:.6f}\n"
            f"=== {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | END | "
            f"{args.experiment_name} ==="
        ),
    )


if __name__ == "__main__":
    main()
