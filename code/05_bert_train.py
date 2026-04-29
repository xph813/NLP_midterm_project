"""
05 BERT 训练（bert-base + roberta-base 对比）

- 从 train.tsv 切 5% 出 dev 用于早停 / 选 best
- 用 HuggingFace Trainer + fp16
- 已经训过的模型（outputs/bert/<name>/done.flag 存在）会跳过
- 训练日志（loss / eval F1）每 step 写到 outputs/bert/<name>/training_log.csv

3090 上每个模型约 5-10 分钟。Mac 上同样能跑（CPU/MPS），只是慢。
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (DATA_CLEAN, LABEL_NAMES, LOG_DIR, NUM_LABELS, OUT,
                   compute_metrics, get_device, get_model_path, save_json,
                   set_seed, setup_hf_mirror, setup_logger)

setup_hf_mirror()

import inspect

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, TrainerCallback)

BERT_DIR = OUT / "bert"


def _trainer_tokenizer_kw():
    """transformers 5.x 把 tokenizer 改成 processing_class，这里自动选。"""
    sig = inspect.signature(Trainer.__init__).parameters
    return "processing_class" if "processing_class" in sig else "tokenizer"


class TitleDataset(Dataset):
    def __init__(self, titles, labels, tokenizer, max_len=64):
        self.titles = list(titles)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.titles[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    m = compute_metrics(labels.tolist(), preds.tolist())
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"]}


class CsvLogCallback(TrainerCallback):
    def __init__(self, save_path):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.save_path.exists():
            with open(self.save_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["step", "epoch", "type", "loss",
                     "eval_accuracy", "eval_macro_f1", "eval_weighted_f1"])

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is None:
            return
        with open(self.save_path, "a", newline="") as f:
            w = csv.writer(f)
            row = [state.global_step, round(state.epoch or 0, 4),
                   "eval" if "eval_loss" in logs else "train",
                   logs.get("loss", logs.get("eval_loss", "")),
                   logs.get("eval_accuracy", ""),
                   logs.get("eval_macro_f1", ""),
                   logs.get("eval_weighted_f1", "")]
            w.writerow(row)


def train_one(model_name: str, save_dir: Path, train_ds, dev_ds, args, logger):
    flag_file = save_dir / "done.flag"
    if flag_file.exists() and not args.force:
        logger.info(f"  {save_dir.name} 已训过（done.flag 存在），跳过")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = get_model_path(model_name, logger=logger)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=NUM_LABELS,
        id2label={i: LABEL_NAMES[i] for i in range(NUM_LABELS)},
        label2id={LABEL_NAMES[i]: i for i in range(NUM_LABELS)})

    use_fp16 = (get_device() == "cuda")
    targs = TrainingArguments(
        output_dir=str(save_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
    )
    tok_kw = _trainer_tokenizer_kw()
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=metrics_fn,
        callbacks=[CsvLogCallback(save_dir / "training_log.csv")],
        **{tok_kw: tokenizer},
    )

    trainer.train()
    trainer.save_model(str(save_dir / "best"))
    tokenizer.save_pretrained(str(save_dir / "best"))
    eval_metrics = trainer.evaluate()
    save_json(eval_metrics, save_dir / "dev_metrics.json")
    flag_file.write_text("done")
    logger.info(f"  {save_dir.name} 训练完成: {eval_metrics}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--models", nargs="+",
                        default=["bert-base-uncased", "roberta-base"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    BERT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("bert", LOG_DIR / "05_bert_train.log")
    logger.info(f"device={get_device()}  models={args.models}")

    train_df = pd.read_csv(DATA_CLEAN / "train.tsv", sep="\t")
    inner_tr, dev_df = train_test_split(
        train_df, test_size=0.05, stratify=train_df["label"],
        random_state=args.seed)
    logger.info(f"inner_train={len(inner_tr)}  dev={len(dev_df)}")

    for model_name in args.models:
        logger.info(f"\n===== 训练 {model_name} =====")
        model_path = get_model_path(model_name, logger=logger)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        train_ds = TitleDataset(inner_tr["title"], inner_tr["label"],
                                tokenizer, args.max_len)
        dev_ds = TitleDataset(dev_df["title"], dev_df["label"],
                              tokenizer, args.max_len)
        save_dir = BERT_DIR / model_name.replace("/", "_")
        train_one(model_name, save_dir, train_ds, dev_ds, args, logger)


if __name__ == "__main__":
    main()
