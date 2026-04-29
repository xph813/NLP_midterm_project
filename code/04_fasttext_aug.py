"""
04 FastText 数据增强对比 + 在 test 上评测

读 03 找到的最优配置，做 4 组对比：
  none    -- 不增强
  drop    -- 随机删词 (p=0.1)
  swap    -- 随机交换相邻词 1 次
  both    -- drop + swap 同时
对每种增强：用 inner_train+dev 全量训练，dev 上看效果。
最后用最佳增强 + 最优超参在全量 train 上重训，在 test 上评测一次。

输出：
  outputs/fasttext/aug_results.csv
  outputs/fasttext/best_model.bin
  outputs/predictions/fasttext_test.tsv
  outputs/metrics/fasttext_test.json
  outputs/figs/07_fasttext_confusion.png
"""
from __future__ import annotations
import argparse
import csv
import random
import sys
from pathlib import Path

import fasttext
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (DATA_CLEAN, LOG_DIR, OUT, compute_metrics, load_json,
                   plot_confusion, save_json, save_predictions, set_seed,
                   setup_logger)
import importlib.util as _u
_p = Path(__file__).resolve().parent / "03_fasttext_grid.py"
_spec = _u.spec_from_file_location("ft_grid", _p)
_m = _u.module_from_spec(_spec); _spec.loader.exec_module(_m)
clean_text = _m.clean_text
to_ft_format = _m.to_ft_format
predict_file = _m.predict_file

FT_DIR = OUT / "fasttext"


def aug_drop(words: list[str], p: float, rng: random.Random) -> list[str]:
    if len(words) <= 3:
        return words
    out = [w for w in words if rng.random() > p]
    return out if out else words


def aug_swap(words: list[str], rng: random.Random) -> list[str]:
    if len(words) < 2:
        return words
    i = rng.randint(0, len(words) - 2)
    words = words.copy()
    words[i], words[i + 1] = words[i + 1], words[i]
    return words


def write_aug_file(df: pd.DataFrame, save_path: Path, mode: str,
                   seed: int = 42) -> None:
    rng = random.Random(seed)
    with open(save_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            title = clean_text(row["title"])
            label = int(row["label"])
            f.write(f"__label__{label} {title}\n")
            if mode == "none":
                continue
            words = title.split()
            if mode in ("drop", "both"):
                aug = " ".join(aug_drop(words, 0.1, rng))
                if aug != title and aug:
                    f.write(f"__label__{label} {aug}\n")
            if mode in ("swap", "both"):
                aug = " ".join(aug_swap(words, rng))
                if aug != title and aug:
                    f.write(f"__label__{label} {aug}\n")


def predict_dataframe(model, df: pd.DataFrame):
    """对 dataframe 推理，返回 (y_true, y_pred, probs)。"""
    y_true = df["label"].astype(int).tolist()
    y_pred, probs = [], []
    for _, row in df.iterrows():
        text = clean_text(row["title"])
        labels, scores = model.predict(text, k=3)
        prob = np.zeros(3, dtype=np.float32)
        for lab, sc in zip(labels, scores):
            prob[int(lab.replace("__label__", ""))] = float(sc)
        y_pred.append(int(np.argmax(prob)))
        probs.append(prob)
    return y_true, y_pred, np.stack(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)

    logger = setup_logger("ft_aug", LOG_DIR / "04_fasttext_aug.log")
    FT_DIR.mkdir(parents=True, exist_ok=True)

    best_cfg = load_json(FT_DIR / "best_config.json")
    lr = best_cfg["lr"]
    epoch = best_cfg["epoch"]
    ngrams = best_cfg["wordNgrams"]
    dim = best_cfg["dim"]
    logger.info(f"最优超参: lr={lr} epoch={epoch} wordNgrams={ngrams} dim={dim}")

    inner_train_df = pd.read_csv(DATA_CLEAN / "train.tsv", sep="\t")
    from sklearn.model_selection import train_test_split
    inner_tr, dev_df = train_test_split(
        inner_train_df, test_size=0.1, stratify=inner_train_df["label"],
        random_state=args.seed)

    dev_path = FT_DIR / "dev.txt"
    if not dev_path.exists():
        to_ft_format(dev_df, dev_path)

    aug_csv = FT_DIR / "aug_results.csv"
    if not aug_csv.exists() or args.force:
        with open(aug_csv, "w", newline="") as f:
            csv.writer(f).writerow(["aug_mode", "dev_acc", "dev_macro_f1"])

    done_modes = set()
    with open(aug_csv) as f:
        for r in csv.DictReader(f):
            done_modes.add(r["aug_mode"])

    fout = open(aug_csv, "a", newline="")
    writer = csv.writer(fout)
    for mode in ["none", "drop", "swap", "both"]:
        if mode in done_modes and not args.force:
            logger.info(f"  跳过 {mode}（已完成）")
            continue
        train_path = FT_DIR / f"aug_train_{mode}.txt"
        write_aug_file(inner_tr, train_path, mode, seed=args.seed)
        logger.info(f"训练 aug={mode}  样本数={sum(1 for _ in open(train_path))}")
        model = fasttext.train_supervised(
            input=str(train_path),
            lr=lr, epoch=epoch, wordNgrams=ngrams,
            dim=dim, loss="softmax", verbose=0, minCount=1)
        y_t, y_p = predict_file(model, dev_path)
        m = compute_metrics(y_t, y_p)
        writer.writerow([mode, f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}"])
        fout.flush()
        logger.info(f"  dev acc={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}")
    fout.close()

    aug_df = pd.read_csv(aug_csv)
    best_aug = aug_df.sort_values("dev_macro_f1", ascending=False).iloc[0]
    best_mode = best_aug["aug_mode"]
    logger.info(f"最佳增强: {best_mode} (dev macro-F1={best_aug['dev_macro_f1']:.4f})")

    full_train_df = pd.read_csv(DATA_CLEAN / "train.tsv", sep="\t")
    test_df = pd.read_csv(DATA_CLEAN / "test.tsv", sep="\t")
    full_train_path = FT_DIR / f"full_train_{best_mode}.txt"
    write_aug_file(full_train_df, full_train_path, best_mode, seed=args.seed)

    final_model_path = FT_DIR / "best_model.bin"
    if final_model_path.exists() and not args.force:
        logger.info(f"加载已存在的最终模型: {final_model_path}")
        model = fasttext.load_model(str(final_model_path))
    else:
        logger.info(f"在全量 train（增强={best_mode}）上重训")
        model = fasttext.train_supervised(
            input=str(full_train_path),
            lr=lr, epoch=epoch, wordNgrams=ngrams,
            dim=dim, loss="softmax", verbose=0, minCount=1)
        model.save_model(str(final_model_path))

    y_true, y_pred, probs = predict_dataframe(model, test_df)
    m = compute_metrics(y_true, y_pred)
    logger.info(f"\n========= FastText 测试集结果 =========\n{m['report']}")

    save_json({**m, "best_config": best_cfg, "best_aug": best_mode},
              OUT / "metrics" / "fasttext_test.json")
    save_predictions(test_df["id"].tolist(), test_df["title"].tolist(),
                     y_true, y_pred,
                     OUT / "predictions" / "fasttext_test.tsv",
                     probs=probs)
    plot_confusion(y_true, y_pred, OUT / "figs" / "07_fasttext_confusion.png",
                   title="FastText – Confusion Matrix")


if __name__ == "__main__":
    main()
