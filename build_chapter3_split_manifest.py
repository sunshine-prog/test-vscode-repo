from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _relative_paths(paths: list[Path], root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed split manifest for chapter 3 experiments")
    parser.add_argument("--data-root", default="dataset_real", help="Dataset root directory")
    parser.add_argument("--output", default="configs/chapter3_split_manifest.json", help="Output manifest path")
    parser.add_argument("--val-defect-count", type=int, default=20, help="Number of defect images reserved for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for manifest generation")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_normal = sorted((data_root / "train" / "normal").glob("*.jpg"))
    val_normal = sorted((data_root / "val" / "normal").glob("*.jpg"))
    test_normal = sorted((data_root / "test" / "normal").glob("*.jpg"))
    test_defect = sorted((data_root / "test" / "defect").glob("*.jpg"))

    if not train_normal or not val_normal or not test_normal or not test_defect:
        raise FileNotFoundError("Dataset structure is incomplete. Expected normal/defect jpg files under train/val/test.")

    val_defect_count = max(1, min(args.val_defect_count, len(test_defect) - 1))
    rng = np.random.default_rng(args.seed)
    chosen_indices = sorted(int(index) for index in rng.choice(len(test_defect), size=val_defect_count, replace=False))
    val_defect = [test_defect[index] for index in chosen_indices]
    val_defect_keys = {path.as_posix() for path in val_defect}
    test_defect_final = [path for path in test_defect if path.as_posix() not in val_defect_keys]

    manifest = {
        "metadata": {
            "data_root": data_root.as_posix(),
            "seed": int(args.seed),
            "val_defect_count": int(val_defect_count),
            "counts": {
                "train_normal": len(train_normal),
                "val_normal": len(val_normal),
                "val_defect": len(val_defect),
                "test_normal": len(test_normal),
                "test_defect": len(test_defect_final),
            },
        },
        "splits": {
            "train": {
                "normal": _relative_paths(train_normal, data_root),
            },
            "val": {
                "normal": _relative_paths(val_normal, data_root),
                "defect": _relative_paths(val_defect, data_root),
            },
            "test": {
                "normal": _relative_paths(test_normal, data_root),
                "defect": _relative_paths(test_defect_final, data_root),
            },
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"manifest={output_path}")
    print(json.dumps(manifest["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
