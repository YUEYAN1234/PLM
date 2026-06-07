from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_from_disk

from .mlm_collator import ProteinMLMCollator
from .modeling import create_mlm_model
from .tokenizer import load_tokenizer


def load_config(path: Path) -> dict[str, object]:
    with path.open("rt", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def precision_flags() -> dict[str, bool]:
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    if torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False}
    return {"bf16": False, "fp16": True}


def training_arguments(output_dir: str, config: dict[str, object]):
    from transformers import TrainingArguments

    training = dict(config)
    params = {
        "output_dir": output_dir,
        "overwrite_output_dir": False,
        "num_train_epochs": float(training.get("num_train_epochs", 1)),
        "max_steps": int(training.get("max_steps", -1)),
        "per_device_train_batch_size": int(training.get("per_device_train_batch_size", 8)),
        "per_device_eval_batch_size": int(training.get("per_device_eval_batch_size", 8)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(training.get("learning_rate", 3e-4)),
        "warmup_ratio": float(training.get("warmup_ratio", 0.05)),
        "weight_decay": float(training.get("weight_decay", 0.01)),
        "lr_scheduler_type": str(training.get("lr_scheduler_type", "cosine")),
        "save_steps": int(training.get("save_steps", 5000)),
        "eval_steps": int(training.get("eval_steps", 5000)),
        "logging_steps": int(training.get("logging_steps", 100)),
        "save_total_limit": int(training.get("save_total_limit", 3)),
        "dataloader_num_workers": int(training.get("dataloader_num_workers", 2)),
        "report_to": [] if str(training.get("report_to", "none")).lower() == "none" else training.get("report_to"),
        "load_best_model_at_end": False,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "do_train": True,
        "do_eval": True,
        **precision_flags(),
    }

    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        params["eval_strategy"] = "steps"
    else:
        params["evaluation_strategy"] = "steps"
    if "save_strategy" in signature:
        params["save_strategy"] = "steps"

    params = {key: value for key, value in params.items() if key in signature}
    return TrainingArguments(**params)


def train(
    *,
    config_path: Path,
    dataset_path_override: Path | None = None,
    output_dir_override: Path | None = None,
    max_steps_override: int | None = None,
    resume_from_checkpoint: str | None = None,
) -> None:
    from transformers import Trainer

    config = load_config(config_path)
    seed = int(config.get("seed", 13))
    set_seed(seed)

    data_config = dict(config.get("data", {}))
    tokenizer_config = dict(config.get("tokenizer", {}))
    model_config = dict(config.get("model", {}))
    train_config = dict(config.get("training", {}))

    dataset_path = Path(dataset_path_override or data_config.get("dataset_path", "data/processed/uniref50_len512"))
    output_dir = Path(output_dir_override or train_config.get("output_dir", "runs/small_esm_mlm"))
    if max_steps_override is not None:
        train_config["max_steps"] = max_steps_override
        train_config["eval_steps"] = min(int(train_config.get("eval_steps", 5000)), max(1, max_steps_override))
        train_config["save_steps"] = min(int(train_config.get("save_steps", 5000)), max(1, max_steps_override))

    dataset = load_from_disk(str(dataset_path))
    tokenizer = load_tokenizer(dataset_path)
    tokenizer.save(output_dir)

    model = create_mlm_model(tokenizer, model_config)
    collator = ProteinMLMCollator(
        tokenizer=tokenizer,
        mlm_probability=float(tokenizer_config.get("mlm_probability", 0.15)),
    )
    args = training_arguments(str(output_dir), train_config)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save(best_dir)
    summary = {
        "dataset_path": str(dataset_path),
        "final_model_dir": str(best_dir),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a small ESM-like protein MLM.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    args = parser.parse_args(argv)

    train(
        config_path=args.config,
        dataset_path_override=args.dataset_path,
        output_dir_override=args.output_dir,
        max_steps_override=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
