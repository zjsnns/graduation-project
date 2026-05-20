param(
    [string]$PythonExe = "D:\graduation project\venv\Scripts\python.exe",
    [string]$DataDir = "D:\graduation project\Natural Gas Dataset\Natural Gas Dataset",
    [string]$ModelDir = "D:\graduation project\Natural Gas Dataset\Natural Gas Dataset\sundial\sundial-base-128m",
    [string]$OutputDir = "D:\graduation project\Natural Gas Dataset\Natural Gas Dataset\sundial\outputs",
    [int]$LookbackLen = 256,
    [int]$ForecastLen = 96,
    [int]$Stride = 16,
    [int]$Epochs = 10,
    [double]$Lr = 0.0002,
    [int]$BatchSize = 2,
    [int]$GradAccum = 8,
    [switch]$WellBalancedSampling,
    [double]$WellBalancePower = 1.0
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$trainScript = Join-Path $scriptDir "train_sundial_lora.py"

Write-Host "[RUN] Starting Sundial LoRA fine-tuning..."
Write-Host "[RUN] DataDir    : $DataDir"
Write-Host "[RUN] ModelDir   : $ModelDir"
Write-Host "[RUN] OutputDir  : $OutputDir"
Write-Host "[RUN] WellBalancedSampling : $WellBalancedSampling"
Write-Host "[RUN] WellBalancePower     : $WellBalancePower"

$extraArgs = @(
  "--well_balance_power", "$WellBalancePower"
)
if ($WellBalancedSampling) {
  $extraArgs += "--well_balanced_sampling"
}

& $PythonExe $trainScript `
  --data_dir "$DataDir" `
  --model_dir "$ModelDir" `
  --output_dir "$OutputDir" `
  --lookback_len $LookbackLen `
  --forecast_len $ForecastLen `
  --stride $Stride `
  --epochs $Epochs `
  --lr $Lr `
  --batch_size $BatchSize `
  --grad_accum $GradAccum `
  @extraArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "[RUN] Fine-tuning failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "[RUN] Fine-tuning finished successfully."
