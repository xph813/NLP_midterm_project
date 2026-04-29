"""
13 Qwen 模型大小对比 (1.5B vs 7B)

用同一批 outputs/qwen/sample_500.tsv 和同一种 prompt（few-shot），
分别跑 Qwen2.5-1.5B-Instruct 和 Qwen2.5-7B-Instruct（7B 直接复用 08 已跑结果），
对比 accuracy / macro-F1 / 推理速度 / 显存占用。

输出：
  outputs/predictions/qwen_1.5B_few.tsv
  outputs/metrics/qwen_size_compare.json
  outputs/tables/qwen_size_compare.csv
  outputs/figs/19_qwen_size_compare.png
"""
from __future__ import annotations
import argparse
import importlib.util as _u
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (LABEL_NAMES, LOG_DIR, OUT, compute_metrics, get_device,
                   load_json, plot_confusion, save_json, set_seed,
                   setup_hf_mirror, setup_logger)
setup_hf_mirror()

QWEN_DIR = OUT / "qwen"
SAMPLE_PATH = QWEN_DIR / "sample_500.tsv"

_spec = _u.spec_from_file_location(
    "qwen_eval", Path(__file__).resolve().parent / "08_qwen_eval.py")
qe = _u.module_from_spec(_spec); _spec.loader.exec_module(qe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--big", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--mode", default="few", choices=["zero", "few", "cot"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)

    for sub in ["tables", "metrics", "predictions", "figs"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    logger = setup_logger("qwen_size", LOG_DIR / "13_qwen_size.log")

    if not SAMPLE_PATH.exists():
        logger.error("缺 sample_500.tsv，请先跑 08_qwen_eval.py")
        return
    sample_df = pd.read_csv(SAMPLE_PATH, sep="\t")
    logger.info(f"评测样本: {len(sample_df)} 条 (mode={args.mode})")

    out_small = OUT / "predictions" / f"qwen_1.5B_{args.mode}.tsv"
    if out_small.exists() and not args.force:
        logger.info(f"复用已有 {out_small.name}")
        df_small = pd.read_csv(out_small, sep="\t")
        time_small = float("nan")
    else:
        logger.info(f"\n===== 加载 1.5B：{args.small} =====")
        tokenizer, model = qe.load_qwen(args.small, logger)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        t0 = time.time()
        df_small = qe.run_mode(args.mode, sample_df, tokenizer, model, logger,
                               args.max_new_tokens, args.bs, args.force)
        time_small = time.time() - t0
        if out_small.name != f"qwen_{args.mode}.tsv":
            df_small.to_csv(out_small, sep="\t", index=False)
        else:
            (OUT / "predictions" / f"qwen_{args.mode}.tsv").rename(out_small)

        del model, tokenizer
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    out_big = OUT / "predictions" / f"qwen_{args.mode}.tsv"
    if not out_big.exists():
        logger.error(f"缺 7B 的 {args.mode} 结果，请先跑 08_qwen_eval.py")
        return
    df_big = pd.read_csv(out_big, sep="\t")
    logger.info(f"7B 结果直接复用 {out_big.name}（来自 08）")

    rows = []
    for tag, df, t in [("Qwen-1.5B", df_small, time_small),
                       ("Qwen-7B", df_big, float("nan"))]:
        valid = df[df["pred_label"] != -1]
        m = compute_metrics(valid["true_label"].astype(int).tolist(),
                            valid["pred_label"].astype(int).tolist())
        rows.append({
            "model": tag,
            "n_valid": len(valid),
            "abstain": int((df["pred_label"] == -1).sum()),
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "f1_Hard": m["per_class"]["Hard"]["f1"],
            "f1_Soft": m["per_class"]["Soft"]["f1"],
            "f1_Other": m["per_class"]["Other"]["f1"],
            "inference_seconds": round(t, 1) if not np.isnan(t) else "",
        })
        plot_confusion(valid["true_label"].astype(int),
                       valid["pred_label"].astype(int),
                       OUT / "figs" / f"19a_{tag}_confusion.png",
                       title=f"{tag} ({args.mode}) – Confusion Matrix")
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT / "tables" / "qwen_size_compare.csv", index=False)
    save_json(rows, OUT / "metrics" / "qwen_size_compare.json")
    logger.info(f"\n大小对比:\n{out_df.to_string(index=False)}")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(out_df))
    w = 0.27
    ax.bar(x - w, out_df["accuracy"], w, label="Accuracy")
    ax.bar(x, out_df["macro_f1"], w, label="Macro F1")
    ax.bar(x + w, out_df["weighted_f1"], w, label="Weighted F1")
    ax.set_xticks(x)
    ax.set_xticklabels(out_df["model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"Qwen size comparison (prompt = {args.mode})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "figs" / "19_qwen_size_compare.png", dpi=150)
    plt.close()
    logger.info("19_qwen_size_compare.png 已存")


if __name__ == "__main__":
    main()
