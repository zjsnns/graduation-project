# 最终报告代码归档说明

归档日期：2026-05-20

归档目录：

```text
D:\graduation project\Natural Gas Dataset\final report\code
```

该目录保存论文最终报告涉及的主要训练代码、推理代码、Transformer 最终权重和 Sundial 对比代码。当前目录不是完整数据包，原始井数据和 Sundial 基座权重需要单独提供。

## 1. 当前目录内容

```text
code
├── README_CODE_20260520.md
├── transformer_final_train.py
├── transformer_final_inference.py
├── run_naive_baselines.py
├── train_lstm_baseline.py
├── train_rf_baseline.py
├── train_cnn_baseline.py
├── final_transformer_model
│   ├── best_model.pth
│   ├── run_config.txt
│   ├── val_per_well_metrics.csv
│   ├── training_history.csv
│   └── well_stats.csv
├── lstm_runs_strict_baseline
└── sundial
    ├── train_sundial_lora.py
    ├── evaluate_all_wells_sundial.py
    ├── sundial_finetune_utils.py
    ├── run_finetune.ps1
    └── sundial-base-128m_code
        ├── configuration_sundial.py
        ├── flow_loss.py
        ├── modeling_sundial.py
        ├── ts_generation_mixin.py
        ├── config.json
        ├── generation_config.json
        └── README.md
```

已删除的非必要内容包括 `inference_outputs`、`inference_outputs_smoke`、`__pycache__`、`final_transformer_model\summary_metrics.csv` 和 `final_transformer_model\test_per_well_metrics.csv`。这些文件不影响最终模型权重和代码复现。

## 2. 原始井数据大小

原始井数据位于：

```text
D:\graduation project\Natural Gas Dataset\Natural Gas Dataset
```

其中 30 个单井 `.xlsx` 文件合计：

```text
文件数：30
总大小：20,763,343 bytes
约等于：19.80 MB
```

如果把同目录下的 `number of samples.xls` 也算入 Excel 文件，则为：

```text
文件数：31
总大小：20,786,895 bytes
约等于：19.82 MB
```

原始井数据体积不大，建议上交时与 `code` 文件夹一起提供。否则别人电脑只能查看代码和权重，不能重新训练或重新推理。

## 3. 各文件用途

| 文件或目录 | 用途 |
|---|---|
| `transformer_final_train.py` | 最终 Transformer 训练脚本 |
| `transformer_final_inference.py` | 加载最终 Transformer 权重并进行测试集推理 |
| `final_transformer_model\best_model.pth` | 最终 Transformer 模型权重 |
| `final_transformer_model\run_config.txt` | 最终 Transformer 训练配置 |
| `final_transformer_model\val_per_well_metrics.csv` | 最终 Transformer 验证集逐井指标 |
| `final_transformer_model\training_history.csv` | 最终 Transformer 训练过程记录 |
| `final_transformer_model\well_stats.csv` | 参与训练与评估的井统计信息 |
| `run_naive_baselines.py` | Naive、Moving Average 等朴素基线 |
| `train_lstm_baseline.py` | LSTM 对照模型 |
| `train_rf_baseline.py` | RandomForest 对照模型 |
| `train_cnn_baseline.py` | CNN 对照模型 |
| `sundial\train_sundial_lora.py` | Sundial LoRA 微调脚本 |
| `sundial\evaluate_all_wells_sundial.py` | Sundial 全井评估脚本 |
| `sundial\sundial_finetune_utils.py` | Sundial 数据清洗、窗口构造和指标函数 |
| `sundial\sundial-base-128m_code` | Sundial 模型结构代码和配置，不包含大权重 |

## 4. 最终 Transformer 模型

当前 `final_transformer_model` 是论文 96 天预测任务下的最终正式 Transformer 模型。

| 项目 | 值 |
|---|---|
| 技术路线 | 多变量 Transformer 直接多步预测 |
| 输入窗口 | 过去 256 天 |
| 预测窗口 | 未来 96 天 |
| 输入特征 | `gas_rate`, `tubing_pressure`, `casing_pressure`, `days_since_open` |
| seed | 44 |
| 权重文件 | `final_transformer_model\best_model.pth` |
| 权重大小 | 约 2.90 MB |

最终测试集指标如下。这些数值来自原始最终运行记录，当前归档目录已不再保留 `summary_metrics.csv` 和 `test_per_well_metrics.csv`。

| 指标口径 | MAE | RMSE | R2 |
|---|---:|---:|---:|
| 逐井平均 | 0.123646 | 0.161460 | -0.216094 |
| pooled points | 0.123646 | 0.178643 | 0.894061 |

说明：逐井平均 R2 是先对每口井单独计算 R2 再取平均，容易受低波动井影响。pooled R2 是将所有井、所有预测点合并后统一计算，更反映整体点级拟合能力，但高波动井权重更大。

## 5. 在别人电脑上能否直接运行

只复制当前 `code` 文件夹，不能保证完整复现所有实验。原因是原始井数据没有放在 `code` 内，Sundial 基座大权重也没有放入归档。

| 内容 | 是否可直接使用 | 条件 |
|---|---|---|
| 查看代码 | 可以 | 不需要额外文件 |
| 加载 Transformer 权重 | 可以 | 需要安装 Python 依赖 |
| Transformer 推理 | 可以 | 需要提供原始 `.xlsx` 数据目录 |
| Transformer 重新训练 | 可以 | 需要提供原始 `.xlsx` 数据目录 |
| LSTM/CNN/RF/Naive 重新训练 | 可以 | 需要提供原始 `.xlsx` 数据目录 |
| Sundial 重新评估 | 不能直接完整运行 | 需要 Sundial 基座权重和 LoRA adapter |
| Sundial 重新微调 | 不能直接完整运行 | 需要 Sundial 基座权重、LoRA 环境和 GPU |

需要额外提供的内容：

| 内容 | 是否已在 `code` 中 | 说明 |
|---|---|---|
| 30 个原始井 `.xlsx` | 否 | 合计约 19.80 MB，建议一并上交 |
| Transformer 权重 | 是 | `final_transformer_model\best_model.pth` |
| Sundial 模型结构代码 | 是 | `sundial\sundial-base-128m_code` |
| Sundial 基座权重 `model.safetensors` | 否 | 约 513 MB，当前未放入代码归档 |
| Sundial LoRA adapter 权重 | 否 | 当前未放入代码归档 |
| Python 依赖清单 | 未单独生成 | 可按第 9 节安装 |

## 6. 推荐上交结构

如果希望别人电脑尽量开箱即跑，建议上交时采用如下结构：

```text
submitted_project
├── code
│   ├── transformer_final_train.py
│   ├── transformer_final_inference.py
│   ├── final_transformer_model
│   ├── run_naive_baselines.py
│   ├── train_lstm_baseline.py
│   ├── train_rf_baseline.py
│   ├── train_cnn_baseline.py
│   └── sundial
└── raw_data
    ├── 54-16X.xlsx
    ├── 54-21X.xlsx
    ├── ...
    └── 60-34H.xlsx
```

其中 `raw_data` 放 30 个原始单井 Excel 文件。

## 7. Transformer 推理命令

在别人电脑上运行时，建议显式指定 `--project_dir`，不要依赖本机默认路径。

假设别人电脑结构为：

```text
E:\submitted_project
├── code
└── raw_data
```

则运行：

```powershell
cd "E:\submitted_project\code"

python transformer_final_inference.py `
  --project_dir "E:\submitted_project\raw_data" `
  --run_dir ".\final_transformer_model" `
  --split test `
  --output_dir ".\inference_outputs"
```

运行后会重新生成：

| 文件 | 说明 |
|---|---|
| `inference_outputs\test_summary_metrics.csv` | 测试集总体指标 |
| `inference_outputs\test_per_well_metrics.csv` | 测试集逐井指标 |
| `inference_outputs\test_prediction_points.csv` | 每口井未来 96 天真实值与预测值 |
| `inference_outputs\plots\*.png` | 预测曲线图 |

## 8. Transformer 重新训练命令

```powershell
cd "E:\submitted_project\code"

python transformer_final_train.py `
  --data_dir "E:\submitted_project\raw_data" `
  --extra_feature days_since_open `
  --seq_len 256 `
  --pred_len 96 `
  --stride 16 `
  --batch_size 16 `
  --epochs 30 `
  --lr 0.0005 `
  --weight_decay 0.0001 `
  --hidden_size 128 `
  --head_hidden_dim 128 `
  --num_layers 3 `
  --nhead 4 `
  --dropout 0.1 `
  --seed 44 `
  --no_show
```

如果电脑没有 GPU，也可以运行，但训练速度会明显变慢。

## 9. Python 环境依赖

建议 Python 版本使用 3.10 到 3.13。核心依赖如下：

```text
numpy
pandas
scikit-learn
matplotlib
openpyxl
torch
```

如果只运行 Transformer、LSTM、CNN、RF 和 Naive 基线，不需要安装 `transformers`、`peft`。

如果运行 Sundial 路线，需要额外依赖：

```text
transformers
peft
safetensors
accelerate
```

Sundial 路线还需要完整的 `sundial-base-128m` 权重目录，当前代码归档中没有包含大权重文件。

## 10. 上交前建议

建议保留：

```text
README_CODE_20260520.md
transformer_final_train.py
transformer_final_inference.py
final_transformer_model
run_naive_baselines.py
train_lstm_baseline.py
train_rf_baseline.py
train_cnn_baseline.py
sundial
```

如果希望进一步精简，可以删除空的 `lstm_runs_strict_baseline` 目录。

## 11. 已完成烟测

此前已经使用归档后的 Transformer 代码和权重执行过测试集推理烟测。烟测输出文件夹已经删除，但结果如下：

| 指标 | 值 |
|---|---:|
| wells | 25 |
| normalized mean MAE | 0.123646 |
| normalized mean RMSE | 0.161460 |
| normalized mean R2 | -0.216094 |
| normalized pooled R2 | 0.894061 |

这说明在本机环境中，归档后的 Transformer 权重、配置和推理脚本可以独立完成最终模型测试集复评。
