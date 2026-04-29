"""
07 BERT 评测：在 test 上跑出每条样本的预测和概率

会自动比较 05 训出的两个模型（bert-base / roberta-base）的 dev macro_f1，
选 dev 上更高的那个去 test 上评测，结果存为 bert_test 的 metrics + predictions。
另一个模型也跑一遍，存为对应文件名（用于报告里的对比表）。
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (DATA_CLEAN, LOG_DIR, OUT, compute_metrics, get_device,
                   load_json, plot_confusion, save_json, save_predictions,
                   set_seed, setup_hf_mirror, setup_logger)
setup_hf_mirror()

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import importlib.util as _u
_spec = _u.spec_from_file_location(
    "bert_train", Path(__file__).resolve().parent / "05_bert_train.py")
_m = _u.module_from_spec(_spec); _spec.loader.exec_module(_m)
TitleDataset = _m.TitleDataset

BERT_DIR = OUT / "bert"


@torch.no_grad()
def predict(model, tokenizer, df, max_len=64, bs=64, device="cuda"):
    model.eval().to(device)
    ds = TitleDataset(df["title"], df["label"], tokenizer, max_len)
    dl = DataLoader(ds, batch_size=bs, shuffle=False)
    all_logits = []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"])
        all_logits.append(out.logits.detach().cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    preds = probs.argmax(axis=-1)
    return preds, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    logger = setup_logger("bert_eval", LOG_DIR / "07_bert_eval.log")
    device = get_device()
    logger.info(f"device={device}")

    test_df = pd.read_csv(DATA_CLEAN / "test.tsv", sep="\t")
    model_dirs = [d for d in BERT_DIR.iterdir()
                  if d.is_dir() and (d / "best").exists()
                  and d.name != "ablation_max_len"]
    if not model_dirs:
        logger.error("没找到任何已训完的 BERT 模型，请先跑 05_bert_train.py")
        return

    dev_scores = {}
    for d in model_dirs:
        dm = d / "dev_metrics.json"
        if dm.exists():
            dev_scores[d.name] = load_json(dm).get("eval_macro_f1", 0)
    if dev_scores:
        best_name = max(dev_scores, key=dev_scores.get)
        logger.info(f"dev macro-F1: {dev_scores}  --> 最优 {best_name}")
    else:
        best_name = model_dirs[0].name

    for d in model_dirs:
        ckpt = d / "best"
        logger.info(f"\n===== 评测 {d.name} =====")
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
        preds, probs = predict(model, tokenizer, test_df,
                               max_len=args.max_len, bs=args.bs, device=device)
        y_true = test_df["label"].astype(int).tolist()
        y_pred = preds.tolist()
        m = compute_metrics(y_true, y_pred)
        logger.info(f"\n{m['report']}")

        tag = d.name
        save_json({**m, "model": tag},
                  OUT / "metrics" / f"bert_{tag}_test.json")
        save_predictions(test_df["id"].tolist(), test_df["title"].tolist(),
                         y_true, y_pred,
                         OUT / "predictions" / f"bert_{tag}_test.tsv",
                         probs=probs)
        plot_confusion(y_true, y_pred,
                       OUT / "figs" / f"09_bert_{tag}_confusion.png",
                       title=f"{tag} – Confusion Matrix")

        if d.name == best_name:
            save_json({**m, "model": tag},
                      OUT / "metrics" / "bert_test.json")
            save_predictions(test_df["id"].tolist(), test_df["title"].tolist(),
                             y_true, y_pred,
                             OUT / "predictions" / "bert_test.tsv",
                             probs=probs)


if __name__ == "__main__":
    main()
