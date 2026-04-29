"""
06 BERT max_len 消融

固定模型 bert-base-uncased、其他超参与 05 一致，扫 max_len = 32 / 64 / 128。
每跑完一组立刻把 dev 上的指标 append 到 csv。

为了节省时间，这里 epoch 设为 2（够看相对差异），需要看绝对值就走 05。
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (DATA_CLEAN, LOG_DIR, OUT, get_device, save_json, set_seed,
                   setup_hf_mirror, setup_logger)
setup_hf_mirror()

from sklearn.model_selection import train_test_split

import importlib.util as _u
_spec = _u.spec_from_file_location(
    "bert_train", Path(__file__).resolve().parent / "05_bert_train.py")
_m = _u.module_from_spec(_spec); _spec.loader.exec_module(_m)
TitleDataset = _m.TitleDataset
metrics_fn = _m.metrics_fn
CsvLogCallback = _m.CsvLogCallback
_trainer_tokenizer_kw = _m._trainer_tokenizer_kw

from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

ABL_DIR = OUT / "bert" / "ablation_max_len"


def run_one(max_len: int, train_df, dev_df, args, logger):
    save_dir = ABL_DIR / f"len_{max_len}"
    flag = save_dir / "done.flag"
    if flag.exists() and not args.force:
        logger.info(f"  max_len={max_len} 已完成，跳过")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3)

    train_ds = TitleDataset(train_df["title"], train_df["label"],
                            tokenizer, max_len)
    dev_ds = TitleDataset(dev_df["title"], dev_df["label"],
                          tokenizer, max_len)

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
    save_json(metrics, save_dir / "dev_metrics.json")
    flag.write_text("done")
    logger.info(f"  max_len={max_len}  dev macro_f1={metrics.get('eval_macro_f1'):.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_lens", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("bert_abl", LOG_DIR / "06_bert_ablation.log")

    train_df = pd.read_csv(DATA_CLEAN / "train.tsv", sep="\t")
    inner_tr, dev_df = train_test_split(
        train_df, test_size=0.05, stratify=train_df["label"],
        random_state=args.seed)

    summary_path = ABL_DIR / "summary.csv"
    if not summary_path.exists():
        with open(summary_path, "w", newline="") as f:
            csv.writer(f).writerow(["max_len", "dev_acc", "dev_macro_f1",
                                    "dev_weighted_f1"])

    for ml in args.max_lens:
        m = run_one(ml, inner_tr, dev_df, args, logger)
        if m is None:
            continue
        with open(summary_path, "a", newline="") as f:
            csv.writer(f).writerow([ml,
                                    f"{m.get('eval_accuracy', 0):.4f}",
                                    f"{m.get('eval_macro_f1', 0):.4f}",
                                    f"{m.get('eval_weighted_f1', 0):.4f}"])

    df = pd.read_csv(summary_path)
    logger.info(f"\n消融汇总:\n{df.to_string(index=False)}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["max_len"], df["dev_macro_f1"], marker="o", label="macro F1")
    ax.plot(df["max_len"], df["dev_weighted_f1"], marker="s", label="weighted F1")
    ax.set_xlabel("max_len")
    ax.set_ylabel("dev F1")
    ax.set_title("BERT max_len ablation")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "figs" / "08_bert_maxlen_ablation.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
