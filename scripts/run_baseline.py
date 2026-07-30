#!/usr/bin/env python3
"""Run a configured multimodal baseline through the shared E-VQA evaluator."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from pprint import pprint
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-edits", type=int, default=1)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--judge-config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    with args.config.resolve().open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or "alg_name" not in config:
        raise ValueError("The config must be a YAML mapping containing alg_name.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "jsonl").mkdir(parents=True, exist_ok=True)
    config["result_dir"] = str(output_dir)
    config["json_dir"] = str(output_dir / "jsonl")
    config["save_path"] = str(output_dir / "adapter")

    if args.data_path:
        config["eval_annotation_path"] = str(args.data_path.resolve())
    if args.image_root:
        root = str(args.image_root.resolve())
        config["coco_image"] = root
        config["rephrase_image"] = root
    if args.judge_config:
        judge_path = args.judge_config.resolve()
        if not judge_path.is_file():
            raise FileNotFoundError(judge_path)
        config["api_key"] = str(judge_path)
        config["real_world_eval"] = True
    else:
        config["api_key"] = None
        config["real_world_eval"] = False

    if args.dry_run:
        pprint(config)
        return

    if args.num_edits < 1:
        raise ValueError("--num-edits must be positive")
    data_path = Path(config["eval_annotation_path"])
    if not data_path.is_file():
        raise FileNotFoundError(
            f"E-VQA annotation file not found: {data_path}. See data/README.md."
        )

    sys.path.insert(0, str(ROOT))
    from easyeditor import MultimodalEditor, VQADataset

    hparams = SimpleNamespace(**config)
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = VQADataset(
        hparams.eval_annotation_path,
        config=hparams,
        size=args.num_edits,
    )
    result = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        copy=True,
        task=f"vqa_{hparams.alg_name.lower()}",
        load_metrics_path=str(
            output_dir / "jsonl" / f"{hparams.alg_name.lower()}_evqa"
        ),
        MMEBench=False,
    )
    pprint(result[0])


if __name__ == "__main__":
    main()
