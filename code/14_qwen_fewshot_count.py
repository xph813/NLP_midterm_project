"""
14 Qwen few-shot 数量对比

用 Qwen-7B 在同一批 sample_500.tsv 上扫 n-shot ∈ {0, 3, 6, 9}：
  - 0-shot：直接复用 outputs/predictions/qwen_zero.tsv
  - 6-shot：直接复用 outputs/predictions/qwen_few.tsv
  - 3 / 9：本脚本新跑

观察 few-shot 数量对 macro-F1 的边际效益。

输出：
  outputs/predictions/qwen_{N}shot.tsv
  outputs/tables/qwen_fewshot_count.csv
  outputs/figs/20_qwen_fewshot_count.png
"""
from __future__ import annotations
import argparse
import importlib.util as _u
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (LABEL_NAMES, LOG_DIR, OUT, compute_metrics, save_json,
                   set_seed, setup_hf_mirror, setup_logger)
setup_hf_mirror()

QWEN_DIR = OUT / "qwen"
SAMPLE_PATH = QWEN_DIR / "sample_500.tsv"

_spec = _u.spec_from_file_location(
    "qwen_eval", Path(__file__).resolve().parent / "08_qwen_eval.py")
qe = _u.module_from_spec(_spec); _spec.loader.exec_module(qe)


# 12 个示例池：每类 4 个，按需取前 N（保持类别均衡）
EXAMPLE_POOL = [
    ("Walmart Slashes Prices on Last-Generation iPads", "Hard"),
    ("Kim Kardashian shares rare photo of her children", "Soft"),
    ("50 Worst Habits For Belly Fat", "Other"),
    ("Israeli sends observers to military drill in Morocco", "Hard"),
    ("LeBron James scores 35 in Lakers comeback win", "Soft"),
    ("How to maintain your motorcycle in winter", "Other"),
    ("How a favela in Rio got its clean water back, for $42300", "Hard"),
    ("Taylor Swift announces a surprise album drop tonight", "Soft"),
    ("Top 10 children's books for ages 5 to 8", "Other"),
    ("Fed signals two rate cuts before the end of the year", "Hard"),
    ("10 best beach destinations for summer travel", "Soft"),
    ("New treatment for type-2 diabetes shows promise in trial", "Other"),
]


def build_prompt_n(title: str, n_shot: int) -> list[dict]:
    """zero (n=0) 时不放 examples；n>0 时放前 n 个 example。"""
    if n_shot <= 0:
        user = (f'Title: "{title}"\n'
                'Answer with only one of: Hard / Soft / Other.\n'
                'Format: "Answer: <label>"')
    else:
        examples = "\n".join([f'Title: "{t}"\nAnswer: {l}'
                              for t, l in EXAMPLE_POOL[:n_shot]])
        user = (f"Here are some examples.\n{examples}\n\n"
                f'Now classify this:\nTitle: "{title}"\n'
                f'Format: "Answer: <label>"')
    return [{"role": "system", "content": qe.SYSTEM_INSTRUCT},
            {"role": "user", "content": user}]


def run_n_shot(n_shot: int, sample_df, tokenizer, model, logger,
               max_new_tokens: int, batch_size: int, force: bool):
    out_path = OUT / "predictions" / f"qwen_{n_shot}shot.tsv"
    if out_path.exists() and not force:
        logger.info(f"  已存在 {out_path.name}，跳过")
        return pd.read_csv(out_path, sep="\t")

    import torch
    titles = sample_df["title"].astype(str).tolist()
    ids = sample_df["id"].tolist()
    y_true = sample_df["label"].astype(int).tolist()
    rows = []
    for s in tqdm(range(0, len(titles), batch_size),
                  desc=f"qwen-{n_shot}shot", ncols=90):
        e = min(s + batch_size, len(titles))
        batch = titles[s:e]
        prompts = [tokenizer.apply_chat_template(
            build_prompt_n(t, n_shot), tokenize=False,
            add_generation_prompt=True) for t in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tok = gen[:, inputs["input_ids"].shape[1]:]
        outs = tokenizer.batch_decode(new_tok, skip_special_tokens=True)
        for i, raw in enumerate(outs):
            pred = qe.extract_label(raw, "few" if n_shot > 0 else "zero")
            rows.append({
                "id": ids[s + i],
                "title": batch[i],
                "true_label": y_true[s + i],
                "pred_label": pred,
                "raw_response": raw.replace("\n", "\\n"),
            })
        if (s // batch_size) % 10 == 0:
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep="\t", index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--shots", nargs="+", type=int, default=[3, 9])
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    for sub in ["tables", "metrics", "predictions", "figs"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    logger = setup_logger("qwen_fs", LOG_DIR / "14_qwen_fewshot.log")

    if not SAMPLE_PATH.exists():
        logger.error("缺 sample_500.tsv，请先跑 08_qwen_eval.py")
        return
    sample_df = pd.read_csv(SAMPLE_PATH, sep="\t")

    need_load = any(
        not (OUT / "predictions" / f"qwen_{s}shot.tsv").exists()
        for s in args.shots)
    tokenizer = model = None
    if need_load:
        logger.info("\n===== 加载 Qwen-7B =====")
        tokenizer, model = qe.load_qwen(args.model, logger)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    for n in args.shots:
        logger.info(f"\n===== 跑 {n}-shot =====")
        run_n_shot(n, sample_df, tokenizer, model, logger,
                   args.max_new_tokens, args.bs, args.force)

    rows = []
    sources = [
        (0, OUT / "predictions" / "qwen_zero.tsv"),
        (3, OUT / "predictions" / "qwen_3shot.tsv"),
        (6, OUT / "predictions" / "qwen_few.tsv"),
        (9, OUT / "predictions" / "qwen_9shot.tsv"),
    ]
    for n, p in sources:
        if not p.exists():
            logger.warning(f"  跳过 {n}-shot（文件 {p.name} 不存在）")
            continue
        df = pd.read_csv(p, sep="\t")
        valid = df[df["pred_label"] != -1]
        if len(valid) == 0:
            continue
        m = compute_metrics(valid["true_label"].astype(int).tolist(),
                            valid["pred_label"].astype(int).tolist())
        rows.append({"n_shot": n,
                     "n_valid": len(valid),
                     "accuracy": m["accuracy"],
                     "macro_f1": m["macro_f1"],
                     "weighted_f1": m["weighted_f1"]})

    out_df = pd.DataFrame(rows).sort_values("n_shot")
    out_df.to_csv(OUT / "tables" / "qwen_fewshot_count.csv", index=False)
    save_json(rows, OUT / "metrics" / "qwen_fewshot_count.json")
    logger.info(f"\nfew-shot 数量对比:\n{out_df.to_string(index=False)}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(out_df["n_shot"], out_df["accuracy"], marker="o", label="Accuracy")
    ax.plot(out_df["n_shot"], out_df["macro_f1"], marker="s", label="Macro F1")
    ax.plot(out_df["n_shot"], out_df["weighted_f1"], marker="^",
            label="Weighted F1")
    ax.set_xlabel("Number of few-shot examples")
    ax.set_ylabel("Score")
    ax.set_title("Qwen-7B: Effect of few-shot count on classification")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "figs" / "20_qwen_fewshot_count.png", dpi=150)
    plt.close()
    logger.info("20_qwen_fewshot_count.png 已存")


if __name__ == "__main__":
    main()
