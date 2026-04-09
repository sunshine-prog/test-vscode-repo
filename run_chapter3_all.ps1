[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$RunFolderName = "",
    [string]$Device = "cuda",
    [switch]$AllowCpuFallback,
    [switch]$IncludeSmokeTest,
    [switch]$SkipLuaeMain,
    [switch]$SkipFullCurve,
    [switch]$SkipAEBaseline,
    [switch]$SkipFeatureBenchmarks,
    [switch]$SkipAugmentationStudy,
    [switch]$SkipPaperAssets,
    [switch]$SkipFinalAssets
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$BaseOutputsRoot = Join-Path $ProjectRoot "outputs"
$DatasetRoot = Join-Path $ProjectRoot "dataset_real"
$ManifestPath = Join-Path $ProjectRoot "configs\chapter3_split_manifest.json"
$LuaeConfig = Join-Path $ProjectRoot "configs\chapter3_luae.yaml"
$FullCurveConfig = Join-Path $ProjectRoot "configs\chapter3_luae_fullcurve.yaml"
$DefaultPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Resolve-PythonExe {
    param([string]$ConfiguredPath)

    if ($ConfiguredPath -and (Test-Path -LiteralPath $ConfiguredPath)) {
        return (Resolve-Path -LiteralPath $ConfiguredPath).Path
    }

    if (Test-Path -LiteralPath $DefaultPython) {
        return (Resolve-Path -LiteralPath $DefaultPython).Path
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand) {
        return $pythonCommand.Source
    }

    throw "Python was not found. Pass -PythonExe explicitly or create .venv first."
}

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Message
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw $Message
    }
}

function Invoke-PythonStep {
    param(
        [int]$Index,
        [int]$Total,
        [string]$Name,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host ("[{0}/{1}] {2}" -f $Index, $Total, $Name) -ForegroundColor Cyan
    Write-Host ("Command: {0} {1}" -f $script:ResolvedPythonExe, ($Arguments -join " ")) -ForegroundColor DarkGray
    & $script:ResolvedPythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw ("Step failed: {0} (exit code {1})" -f $Name, $LASTEXITCODE)
    }
}

function Test-CudaAvailability {
    param(
        [string]$RequestedDevice,
        [switch]$EnableCpuFallback
    )

    if ($RequestedDevice -ne "cuda") {
        return $RequestedDevice
    }

    $gpuCheckCode = 'import sys, torch; ok=torch.cuda.is_available(); print(''cuda_available='' + str(ok)); print(''gpu='' + (torch.cuda.get_device_name(0) if ok else ''None'')); sys.exit(0 if ok else 2)'
    $gpuCheckOutput = & $script:ResolvedPythonExe -c $gpuCheckCode 2>&1
    $exitCode = $LASTEXITCODE
    foreach ($line in $gpuCheckOutput) {
        Write-Host $line
    }
    if ($exitCode -eq 0) {
        return "cuda"
    }

    if ($EnableCpuFallback) {
        Write-Warning "CUDA is unavailable. Falling back to CPU because -AllowCpuFallback was provided."
        return "cpu"
    }

    throw "CUDA is unavailable. Install GPU-enabled PyTorch or rerun with -Device cpu / -AllowCpuFallback."
}

function Show-WeightHints {
    $padimWeightPath = Join-Path $HOME ".cache\torch\hub\checkpoints\wide_resnet50_2-95faca4d.pth"
    if (-not (Test-Path -LiteralPath $padimWeightPath)) {
        Write-Warning "PaDiM expects local wide_resnet50_2 weights at:"
        Write-Warning ("  {0}" -f $padimWeightPath)
    }
}

$ResolvedPythonExe = Resolve-PythonExe -ConfiguredPath $PythonExe
Assert-PathExists -Path $ProjectRoot -Message "Project root does not exist."
Assert-PathExists -Path $DatasetRoot -Message ("Dataset not found: {0}" -f $DatasetRoot)
Assert-PathExists -Path $ManifestPath -Message ("Split manifest not found: {0}" -f $ManifestPath)
Assert-PathExists -Path $LuaeConfig -Message ("Config not found: {0}" -f $LuaeConfig)
Assert-PathExists -Path $FullCurveConfig -Message ("Config not found: {0}" -f $FullCurveConfig)

$ResolvedDevice = Test-CudaAvailability -RequestedDevice $Device -EnableCpuFallback:$AllowCpuFallback
if ([string]::IsNullOrWhiteSpace($RunFolderName)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunFolderName = "chapter3_repro_{0}_{1}" -f $ResolvedDevice, $timestamp
}
$OutputsRoot = Join-Path $BaseOutputsRoot $RunFolderName
[void](New-Item -ItemType Directory -Path $OutputsRoot -Force)

if (-not $SkipFeatureBenchmarks) {
    Show-WeightHints
}

$steps = New-Object System.Collections.Generic.List[object]

if ($IncludeSmokeTest) {
    $steps.Add([pscustomobject]@{
        Name = "Smoke test"
        Args = @(
            "train_chapter3.py",
            "--config", $LuaeConfig,
            "--device", $ResolvedDevice,
            "--epochs", "3",
            "--max-test-samples", "16",
            "--output-root", (Join-Path $OutputsRoot "chapter3_smoke")
        )
    })
}

if (-not $SkipLuaeMain) {
    $steps.Add([pscustomobject]@{
        Name = "LUAE main result"
        Args = @(
            "train_chapter3.py",
            "--config", $LuaeConfig,
            "--device", $ResolvedDevice,
            "--output-root", (Join-Path $OutputsRoot "chapter3_repro")
        )
    })
}

if (-not $SkipFullCurve) {
    $steps.Add([pscustomobject]@{
        Name = "LUAE full training curve"
        Args = @(
            "train_chapter3.py",
            "--config", $FullCurveConfig,
            "--device", $ResolvedDevice,
            "--output-root", (Join-Path $OutputsRoot "chapter3_fullcurve_run")
        )
    })
}

if (-not $SkipAEBaseline) {
    $steps.Add([pscustomobject]@{
        Name = "AE baseline"
        Args = @(
            "train_chapter3.py",
            "--config", $LuaeConfig,
            "--device", $ResolvedDevice,
            "--pipeline", "base",
            "--output-root", (Join-Path $OutputsRoot "baseline_p768_b32_lr5e4_cosine")
        )
    })
}

if (-not $SkipFeatureBenchmarks) {
    $steps.Add([pscustomobject]@{
        Name = "Feature benchmarks (ResNet18 + PaDiM)"
        Args = @(
            "run_chapter3_feature_benchmarks.py",
            "--config", $LuaeConfig,
            "--method", "both",
            "--output-root", (Join-Path $OutputsRoot "chapter3_feature_benchmarks_final")
        )
    })
}

if (-not $SkipAugmentationStudy) {
    $steps.Add([pscustomobject]@{
        Name = "Augmentation study"
        Args = @(
            "run_chapter3_augmentation_study.py",
            "--config", $LuaeConfig,
            "--device", $ResolvedDevice,
            "--output-root", (Join-Path $OutputsRoot "chapter3_augmentation_study")
        )
    })
}

if (-not $SkipPaperAssets) {
    $steps.Add([pscustomobject]@{
        Name = "Generate LUAE paper assets"
        Args = @(
            "generate_chapter3_paper_assets.py",
            "--outputs-root", $OutputsRoot,
            "--luae-run", "chapter3_repro",
            "--padim-metrics", (Join-Path $OutputsRoot "chapter3_feature_benchmarks_final\padim\metrics\test_metrics.json"),
            "--resnet18-metrics", (Join-Path $OutputsRoot "chapter3_feature_benchmarks_final\resnet18\metrics\test_metrics.json")
        )
    })
    $steps.Add([pscustomobject]@{
        Name = "Render training curve assets"
        Args = @(
            "render_chapter3_training_curve_assets.py",
            "--history-json", (Join-Path $OutputsRoot "chapter3_fullcurve_run\metrics\training_history.json"),
            "--output-dir", (Join-Path $OutputsRoot "chapter3_fullcurve_run\paper_assets")
        )
    })
    $steps.Add([pscustomobject]@{
        Name = "Render augmentation paper assets"
        Args = @(
            "render_chapter3_augmentation_paper_assets.py",
            "--study-root", (Join-Path $OutputsRoot "chapter3_augmentation_study"),
            "--output-dir", (Join-Path $OutputsRoot "chapter3_augmentation_study\paper_assets")
        )
    })
}

if (-not $SkipFinalAssets) {
    $steps.Add([pscustomobject]@{
        Name = "Build final chapter 3 paper assets"
        Args = @(
            "build_chapter3_final_paper_assets.py",
            "--outputs-root", $OutputsRoot
        )
    })
}

Write-Host "==========================================" -ForegroundColor Green
Write-Host " Chapter 3 One-Click Reproduction Script " -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ("Project root : {0}" -f $ProjectRoot)
Write-Host ("Python       : {0}" -f $ResolvedPythonExe)
Write-Host ("Device       : {0}" -f $ResolvedDevice)
Write-Host ("Run folder   : {0}" -f $RunFolderName)
Write-Host ("Outputs root : {0}" -f $OutputsRoot)
Write-Host ("Steps        : {0}" -f $steps.Count)

Set-Location -LiteralPath $ProjectRoot

for ($i = 0; $i -lt $steps.Count; $i++) {
    $step = $steps[$i]
    Invoke-PythonStep -Index ($i + 1) -Total $steps.Count -Name $step.Name -Arguments $step.Args
}

Write-Host ""
Write-Host "All requested Chapter 3 steps finished successfully." -ForegroundColor Green
Write-Host ("Main result directory : {0}" -f (Join-Path $OutputsRoot "chapter3_repro"))
Write-Host ("All outputs root      : {0}" -f $OutputsRoot)
Write-Host ""
Write-Host "Useful examples:" -ForegroundColor Yellow
Write-Host ("  powershell -ExecutionPolicy Bypass -File ""{0}""" -f (Join-Path $ProjectRoot "run_chapter3_all.ps1"))
Write-Host ("  powershell -ExecutionPolicy Bypass -File ""{0}"" -RunFolderName chapter3_gpu_run_01" -f (Join-Path $ProjectRoot "run_chapter3_all.ps1"))
Write-Host ("  powershell -ExecutionPolicy Bypass -File ""{0}"" -IncludeSmokeTest -RunFolderName chapter3_gpu_smoke_and_full" -f (Join-Path $ProjectRoot "run_chapter3_all.ps1"))
Write-Host ("  powershell -ExecutionPolicy Bypass -File ""{0}"" -SkipAugmentationStudy -RunFolderName chapter3_gpu_no_aug" -f (Join-Path $ProjectRoot "run_chapter3_all.ps1"))
