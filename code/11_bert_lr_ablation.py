"""
11 BERT learning rate 消融

固定 bert-base-uncased、max_len=64、epoch=2，扫 lr ∈ {1e-5, 2e-5, 3e-5, 5e-5}。
看哪个 lr 在 dev 上最优。每组训练完立即落盘 done.flag，断点续跑。

输出：
  outputs/bert/ablation_lr/summary.csv
  outputs/figs/17_bert_lr_ablation.png
"""
from __future__ import annotations
import argparse
import csv
import importlib.util as _u
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (DATA_CLEAN, LOG_DIR, OUT, get_device, get_model_path,
                   save_json, set_seed, setup_hf_mirror, setup_logger)
setup_hf_mirror()

from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

_spec = _u.spec_from_file_location(
    "bert_train", Path(__file__).resolve().parent / "05_bert_train.py")
_m = _u.module_from_spec(_spec); _spec.loader.exec_module(_m)
TitleDataset = _m.TitleDataset
metrics_fn = _m.metrics_fn
CsvLogCallback = _m.CsvLogCallback
_trainer_tokenizer_kw = _m._trainer_tokenizer_kw

LR_DIR = OUT / "bert" / "ablation_lr"


def run_one(lr, train_df, dev_df, args, logger):
    save_dir = LR_DIR / f"lr_{lr:.0e}"
    flag = save_dir / "done.flag"
    if flag.exists() and not args.force:
        logger.info(f"  lr={lr} 已完成，跳过")
        from utils import load_json
        return load_json(save_dir / "dev_metrics.json")

    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = get_model_path(args.model, logger=logger)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=3)
    train_ds = TitleDataset(train_df["title"], train_df["label"],
                            tokenizer, args.max_len)
    dev_ds = TitleDataset(dev_df["title"], dev_df["label"],
                          tokenizer, args.max_len)

    targs = TrainingArguments(
        output_dir=str(save_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=(get_device() == "cuda"),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=200,
        report_to=[],
        seed=args.seed,
    )
    tok_kw = _trainer_tokenizer_kw()
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=dev_ds,
        compute_metrics=metrics_fn,
        callbacks=[CsvLogCallback(save_dir / "training_log.csv")],
        **{tok_kw: tokenizer},
    )
    trainer.train()
    metrics = trainer.evaluate()
    save_json(metrics, save_dir / "dev_metrics.json")
    flag.write_text("done")
    logger.info(f"  lr={lr}  dev macro_f1={metrics.get('eval_macro_f1', 0):.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--lrs", nargs="+", type=float,
                        default=[1e-5, 2e-5, 3e-5, 5e-5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    LR_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    logger = setup_logger("bert_lr", LOG_DIR / "11_bert_lr.log")

    train_df = pd.read_csv(DATA_CLEAN / "train.tsv", sep="\t")
    inner_tr, dev_df = train_test_split(
        train_df, test_size=0.05, stratify=train_df["label"],
        random_state=args.seed)

    summary_path = LR_DIR / "summary.csv"
    if not summary_path.exists():
        with open(summary_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["lr", "dev_acc", "dev_macro_f1", "dev_weighted_f1"])

    seen = set()
    if summary_path.exists():
        seen = set(pd.read_csv(summary_path)["lr"].tolist())

    for lr in args.lrs:
        m = run_one(lr, inner_tr, dev_df, args, logger)
        if m is None or lr in seen:
            continue
        with open(summary_path, "a", newline="") as f:
            csv.writer(f).writerow([
                lr,
                f"{m.get('eval_accuracy', 0):.4f}",
                f"{m.get('eval_macro_f1', 0):.4f}",
                f"{m.get('eval_weighted_f1', 0):.4f}",
            ])

    df = pd.read_csv(summary_path).sort_values("lr")
    logger.info(f"\nlr 消融汇总:\n{df.to_string(index=False)}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["lr"], df["dev_macro_f1"], marker="o", label="macro F1")
    ax.plot(df["lr"], df["dev_weighted_f1"], marker="s", label="weighted F1")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("dev F1")
    ax.set_title("BERT learning rate ablation")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "figs" / "17_bert_lr_ablation.png", dpi=150)
    plt.close()
    logger.info("17_bert_lr_ablation.png 已存")


if __name__ == "__main__":
    main()
