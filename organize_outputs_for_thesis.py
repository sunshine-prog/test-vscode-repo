from __future__ import annotations

import shutil
from pathlib import Path


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_dir(src: Path, dst: Path, *, include_names: set[str] | None = None) -> None:
    if not src.exists():
        return
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        relative = item.relative_to(src)
        if "patch_cache" in relative.parts:
            continue
        if include_names is not None and relative.parts and relative.parts[0] not in include_names:
            continue
        target = dst / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def _copy_seed_metrics_and_figures(src_root: Path, dst_root: Path) -> None:
    if not src_root.exists():
        return
    for seed_dir in sorted(src_root.glob("*/seed_*")):
        for subdir_name in ("metrics", "figures"):
            subdir = seed_dir / subdir_name
            if subdir.exists():
                _copy_dir(subdir, dst_root / seed_dir.parent.name / seed_dir.name / subdir_name)


def main() -> None:
    outputs_root = Path(r"D:\pythonProject2\outputs")
    curated_root = outputs_root / "00_论文总目录_最终版"
    if curated_root.exists():
        shutil.rmtree(curated_root, ignore_errors=True)
    curated_root.mkdir(parents=True, exist_ok=True)

    user_root = curated_root / "01_用户训练结果" / "第三章_LUAE"
    codex_root = curated_root / "02_Codex辅助实验"
    paper_root = curated_root / "03_第三章论文输出"
    chapter4_root = curated_root / "04_第四章输出"

    # User training results
    _copy_dir(outputs_root / "chapter3_repro", user_root / "chapter3_repro", include_names={"checkpoints", "figures", "metrics", "paper_assets"})
    _copy_dir(outputs_root / "chapter3_fullcurve_run", user_root / "chapter3_fullcurve_run", include_names={"checkpoints", "figures", "metrics", "paper_assets"})

    # Codex-assisted experiments
    _copy_dir(outputs_root / "chapter3_feature_benchmarks_final", codex_root / "第三章_对比方法", include_names={"padim", "resnet18"})
    for summary_file in (
        outputs_root / "chapter3_augmentation_study" / "augmentation_multiseed_results.csv",
        outputs_root / "chapter3_augmentation_study" / "augmentation_summary.csv",
        outputs_root / "chapter3_augmentation_study" / "augmentation_roc_stats.json",
    ):
        if summary_file.exists():
            _copy_file(summary_file, codex_root / "第三章_增强消融原始结果" / summary_file.name)
    _copy_seed_metrics_and_figures(outputs_root / "chapter3_augmentation_study", codex_root / "第三章_增强消融原始结果")

    # Paper-ready assets
    _copy_dir(outputs_root / "chapter3_repro" / "paper_assets", paper_root / "01_定量对比与残差图")
    _copy_dir(outputs_root / "chapter3_fullcurve_run" / "paper_assets", paper_root / "00_LUAE训练曲线")
    _copy_dir(outputs_root / "chapter3_augmentation_study" / "paper_assets", paper_root / "02_增强消融ROC")
    _copy_dir(outputs_root / "chapter3_feature_benchmarks_final", paper_root / "03_对比方法原始指标", include_names={"padim", "resnet18"})
    _copy_dir(outputs_root / "baseline_p768_b32_lr5e4_cosine", paper_root / "03_对比方法原始指标" / "ae", include_names={"metrics"})

    # Chapter 4 placeholder
    chapter4_root.mkdir(parents=True, exist_ok=True)
    (chapter4_root / "README.md").write_text(
        "# 第四章输出目录\n\n当前目录预留给第四章闭环控制与后续实验结果。\n",
        encoding="utf-8",
    )

    index_path = curated_root / "README_目录说明.md"
    index_path.write_text(
        "\n".join(
            [
                "# 论文输出目录说明",
                "",
                "## 1. 用户训练结果",
                "- `01_用户训练结果/第三章_LUAE/chapter3_repro`：当前论文主结果，对应 LUAE 最终测试指标。",
                "- `01_用户训练结果/第三章_LUAE/chapter3_fullcurve_run`：更长训练曲线的完整重跑结果。",
                "",
                "## 2. Codex辅助实验",
                "- `02_Codex辅助实验/第三章_对比方法`：AE/PaDiM/ResNet18 对比实验结果。",
                "- `02_Codex辅助实验/第三章_增强消融原始结果`：四种增强方式多轮实验原始结果。",
                "",
                "## 3. 第三章论文输出",
                "- `03_第三章论文输出/01_定量对比与残差图`：第三章主表格、柱状图、折线图、残差图。",
                "- `03_第三章论文输出/02_增强消融ROC`：四种增强方式的平滑ROC阴影图、AUC统计和实验说明。",
                "- `03_第三章论文输出/03_对比方法原始指标`：AE/PaDiM/ResNet18 原始指标来源。",
                "",
                "## 4. 第四章输出",
                "- `04_第四章输出`：预留给第四章结果。",
            ]
        ),
        encoding="utf-8",
    )

    print(f"curated_root={curated_root}")
    print(f"index={index_path}")


if __name__ == "__main__":
    main()
