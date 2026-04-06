from __future__ import annotations

import argparse
import copy

from spray_defect.config import load_yaml
from spray_defect.trainer import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="第三章：LUAE 无监督喷涂缺陷检测训练与评估")
    parser.add_argument("--config", default="configs/chapter3_luae.yaml", help="第三章配置文件路径")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖批量大小")
    parser.add_argument("--device", default=None, help="覆盖训练设备，如 cpu / cuda / auto")
    parser.add_argument("--max-test-samples", type=int, default=None, help="只取部分测试样本做快速验证")
    args = parser.parse_args()

    config = copy.deepcopy(load_yaml(args.config))

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["training"]["device"] = args.device

    result = train_and_evaluate(config, max_test_samples=args.max_test_samples)
    metrics = result["metrics"]

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