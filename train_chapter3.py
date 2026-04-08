from __future__ import annotations

import argparse
import copy

from spray_defect.config import load_yaml
from spray_defect.trainer import train_and_evaluate
from spray_defect.trainer_v2 import train_and_evaluate_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="第三章：LUAE 无监督喷涂缺陷检测训练与评估")
    parser.add_argument("--config", default="configs/chapter3_luae.yaml", help="第三章配置文件路径")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖批量大小")
    parser.add_argument("--device", default=None, help="覆盖训练设备，如 cpu / cuda / auto")
    parser.add_argument("--max-test-samples", type=int, default=None, help="只取部分测试样本做快速验证")
    parser.add_argument(
        "--pipeline",
        choices=("auto", "base", "patch"),
        default="auto",
        help="choose the base image pipeline or the patch-aware pipeline",
    )
    parser.add_argument("--seed", type=int, default=None, help="override the random seed")
    parser.add_argument("--output-root", default=None, help="override the output directory")
    parser.add_argument("--deterministic", action="store_true", help="enable deterministic training behavior")
    parser.add_argument("--learning-rate", type=float, default=None, help="override the learning rate")
    parser.add_argument("--norm-type", default=None, help="override model normalization type")
    parser.add_argument("--group-count", type=int, default=None, help="override group count for group normalization")
    parser.add_argument("--scheduler-type", default=None, help="override scheduler type")
    args = parser.parse_args()

    config = copy.deepcopy(load_yaml(args.config))


    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_root is not None:
        config["paths"]["output_root"] = args.output_root
    if args.deterministic:
        config.setdefault("reproducibility", {})["deterministic"] = True
    if args.norm_type is not None:
        config.setdefault("model", {})["norm_type"] = args.norm_type
    if args.group_count is not None:
        config.setdefault("model", {})["group_count"] = args.group_count
    if args.scheduler_type is not None:
        config.setdefault("training", {}).setdefault("scheduler", {})["type"] = args.scheduler_type

    patching_enabled = bool(config.get("patching", {}).get("enabled", False))
    use_patch_pipeline = args.pipeline == "patch" or (args.pipeline == "auto" and patching_enabled)
    train_fn = train_and_evaluate_v2 if use_patch_pipeline else train_and_evaluate

    result = train_fn(config, max_test_samples=args.max_test_samples)
    metrics = result["metrics"]
    print(f"Pipeline: {'patch' if use_patch_pipeline else 'base'}")

    print("\n第三章训练完成。")
    print(f"模型权重: {result['checkpoint_path']}")
    print(
        "测试结果: "
        f"Accuracy={metrics['accuracy']:.4f}, "
        f"Precision={metrics['precision']:.4f}, "
        f"Recall={metrics['recall']:.4f}, "
        f"F1={metrics['f1_score']:.4f}, "
        f"AUC={metrics['auc']:.4f}"
    )
    print(f"阈值: {metrics['threshold']:.6f}")
    print(f"平均单样本推理时延: {metrics['avg_inference_latency_ms']:.2f} ms")


if __name__ == "__main__":
    main()
