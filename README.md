该目录保存论文最终报告涉及的主要训练代码、推理代码、Transformer 最终权重和 Sundial 对比代码。当前目录不是完整数据包，原始井数据和 Sundial 基座权重需要单独下载。

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



## 2. 各文件用途

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

## 3. 最终 Transformer 模型

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




## 4. Transformer 推理命令

运行时，建议显式指定 `--project_dir`，不要依赖本机默认路径。

假设电脑结构为：

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

## 5. Transformer 重新训练命令

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

## 6. Python 环境依赖

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

Sundial 路线还需要完整的 `sundial-base-128m` 权重目录。
