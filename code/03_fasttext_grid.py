"""
03 FastText 网格搜索

把 train.tsv 转 fasttext 格式，从中再切 9:1 的 dev，
跑 lr × epoch × wordNgrams = 18 组，每组结果立刻 append 到 csv，
最后选 dev macro-F1 最高的配置存为 best_config.json。

每组结果会 append 到 grid_results.csv，重启时已有的会跳过。
"""
from __future__ import annotations
import argparse
import csv
import re
import sys
from pathlib import Path

import fasttext
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (DATA_CLEAN, LOG_DIR, OUT, compute_metrics, save_json,
                   set_seed, setup_logger)

FT_DIR = OUT / "fasttext"


def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_ft_format(df: pd.DataFrame, save_path: Path) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(f"__label__{int(row['label'])} {clean_text(row['title'])}\n")


def predict_file(model, path: Path):
    """逐行读 fasttext 文件，返回 (y_true, y_pred)。"""
    y_true, y_pred = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            label_part, _, text = line.partition(" ")
            y_true.append(int(label_part.replace("__label__", "")))
            pred_label = model.predict(text)[0][0].replace("__label__", "")
            y_pred.append(int(pred_label))
    return y_true, y_pred


def plot_heatmap(df_grid: pd.DataFrame, save_path: Path, fix_ngrams: int) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sub = df_grid[df_grid["wordNgrams"] == fix_ngrams]
    pivot = sub.pivot(index="lr", columns="epoch", values="dev_macro_f1")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax)
    ax.set_title(f"FastText dev macro-F1  (wordNgrams={fix_ngrams})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)

    FT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("ft_grid", LOG_DIR / "03_fasttext_grid.log")

    inner_train_path = FT_DIR / "inner_train.txt"
    dev_path = FT_DIR / "dev.txt"
    if args.force or not (inner_train_path.exists() and dev_path.exists()):
        train_df = pd.read_csv(DATA_CLEAN / "train.tsv", sep="\t")
        inner_train, dev = train_test_split(
            train_df, test_size=0.1, stratify=train_df["label"],
            random_state=args.seed)
        to_ft_format(inner_train, inner_train_path)
        to_ft_format(dev, dev_path)
        logger.info(f"inner_train={len(inner_train)}  dev={len(dev)}")

    grid = [(lr, ep, ng)
            for lr in [0.3, 0.5, 1.0]
            for ep in [10, 20, 30]
            for ng in [1, 2]]

    csv_path = FT_DIR / "grid_results.csv"
    done = set()
    if csv_path.exists() and not args.force:
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                done.add((float(r["lr"]), int(r["epoch"]), int(r["wordNgrams"])))
        logger.info(f"已完成 {len(done)} 组，将跳过")
    else:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["lr", "epoch", "wordNgrams", "dim",
                                    "dev_acc", "dev_macro_f1"])

    fout = open(csv_path, "a", newline="")
    writer = csv.writer(fout)
    for (lr, ep, ng) in grid:
        if (lr, ep, ng) in done:
            continue
        logger.info(f"训练 lr={lr} epoch={ep} wordNgrams={ng}")
        model = fasttext.train_supervised(
            input=str(inner_train_path),
            lr=lr, epoch=ep, wordNgrams=ng,
            dim=100, loss="softmax", verbose=0, minCount=1)
        y_true, y_pred = predict_file(model, dev_path)
        m = compute_metrics(y_true, y_pred)
        writer.writerow([lr, ep, ng, 100,
                         f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}"])
        fout.flush()
        logger.info(f"  acc={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}")
    fout.close()

    df = pd.read_csv(csv_path)
    best = df.sort_values("dev_macro_f1", ascending=False).iloc[0].to_dict()
    logger.info(f"最优配置 (按 dev macro-F1): {best}")
    save_json({"lr": float(best["lr"]),
               "epoch": int(best["epoch"]),
               "wordNgrams": int(best["wordNgrams"]),
               "dim": int(best["dim"]),
               "dev_macro_f1": float(best["dev_macro_f1"])},
              FT_DIR / "best_config.json")

    plot_heatmap(df, FT_DIR / "grid_heatmap.png", int(best["wordNgrams"]))
    logger.info(f"热力图: {FT_DIR / 'grid_heatmap.png'}")


if __name__ == "__main__":
    main()
