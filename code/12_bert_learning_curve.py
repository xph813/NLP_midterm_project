"""
12 BERT 训练数据量 learning curve

固定 bert-base-uncased、lr=2e-5、max_len=64、epoch=2，
扫训练集采样比例 ∈ {0.1, 0.3, 0.6, 1.0}，
看 dev macro-F1 怎么随数据量变化。

输出：
  outputs/bert/learning_curve/summary.csv
  outputs/figs/18_bert_learning_curve.png
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
                   load_json, save_json, set_seed, setup_hf_mirror,
                   setup_logger)
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

LC_DIR = OUT / "bert" / "learning_curve"


def run_one(frac, train_df, dev_df, args, logger):
    save_dir = LC_DIR / f"frac_{int(frac * 100):03d}"
    flag = save_dir / "done.flag"
    if flag.exists() and not args.force:
        logger.info(f"  frac={frac} 已完成，跳过")
        return load_json(save_dir / "dev_metrics.json")

    save_dir.mkdir(parents=True, exist_ok=True)

    if frac < 1.0:
        sub_df, _ = train_test_split(
            train_df, train_size=frac, stratify=train_df["label"],
            random_state=args.seed)
    else:
        sub_df = train_df
    logger.info(f"  frac={frac}  使用 {len(sub_df)} 条训练样本")

    model_path = get_model_path(args.model, logger=logger)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=3)
    train_ds = TitleDataset(sub_df["title"], sub_df["label"],
                            tokenizer, args.max_len)
    dev_ds = TitleDataset(dev_df["title"], dev_df["label"],
                          tokenizer, args.max_len)

    targs = TrainingArguments(
        output_dir=str(save_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,
        learning_rate=args.lr,
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
    metrics["n_samples"] = len(sub_df)
    save_json(metrics, save_dir / "dev_metrics.json")
    flag.write_text("done")
    logger.info(f"  frac={frac}  n={len(sub_df)}  "
                f"dev macro_f1={metrics.get('eval_macro_f1', 0):.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--fracs", nargs="+", type=float,
                        default=[0.1, 0.3, 0.6, 1.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    LC_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    logger = setup_logger("bert_lc", LOG_DIR / "12_bert_learning_curve.log")

    train_df = pd.read_csv(DATA_CLEAN / "train.tsv", sep="\t")
    inner_tr, dev_df = train_test_split(
        train_df, test_size=0.05, stratify=train_df["label"],
        random_state=args.seed)

    summary_path = LC_DIR / "summary.csv"
    if not summary_path.exists():
        with open(summary_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["frac", "n_samples", "dev_acc", "dev_macro_f1",
                 "dev_weighted_f1"])

    seen = set()
    if summary_path.exists():
        seen = set(pd.read_csv(summary_path)["frac"].tolist())

    for frac in args.fracs:
        m = run_one(frac, inner_tr, dev_df, args, logger)
        if m is None or frac in seen:
            continue
        with open(summary_path, "a", newline="") as f:
            csv.writer(f).writerow([
                frac, m.get("n_samples", -1),
                f"{m.get('eval_accuracy', 0):.4f}",
                f"{m.get('eval_macro_f1', 0):.4f}",
                f"{m.get('eval_weighted_f1', 0):.4f}",
            ])

    df = pd.read_csv(summary_path).sort_values("frac")
    logger.info(f"\n数据量曲线汇总:\n{df.to_string(index=False)}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["n_samples"], df["dev_macro_f1"], marker="o", label="macro F1")
    ax.plot(df["n_samples"], df["dev_weighted_f1"], marker="s",
            label="weighted F1")
    ax.set_xscale("log")
    ax.set_xlabel("Training samples (log)")
    ax.set_ylabel("dev F1")
    ax.set_title("BERT learning curve (training set size vs F1)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "figs" / "18_bert_learning_curve.png", dpi=150)
    plt.close()
    logger.info("18_bert_learning_curve.png 已存")


if __name__ == "__main__":
    main()
