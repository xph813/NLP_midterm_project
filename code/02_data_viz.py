"""
02 数据可视化（6 张图，给报告第 3 章用）

读 data_clean/clean.tsv + train.tsv + test.tsv，产出：
  outputs/figs/01_label_dist.png       三类标签分布
  outputs/figs/02_source_label.png     MIND/GOOGLE × 标签 堆叠柱
  outputs/figs/03_title_length.png     标题长度直方图（按类着色）
  outputs/figs/04_top_words.png        每类 Top-15 词（横向柱状）
  outputs/figs/05_train_test.png       train/test 类别比例对照
  outputs/figs/06_length_box.png       每类标题长度箱线图
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_CLEAN, LABEL_NAMES, LOG_DIR, OUT, setup_logger

FIG = OUT / "figs"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("viz", LOG_DIR / "02_viz.log")

    df = pd.read_csv(DATA_CLEAN / "clean.tsv", sep="\t")
    train_df = pd.read_csv(DATA_CLEAN / "train.tsv", sep="\t")
    test_df = pd.read_csv(DATA_CLEAN / "test.tsv", sep="\t")

    sns.set_theme(style="whitegrid", context="talk")
    palette = {0: "#d62728", 1: "#2ca02c", 2: "#1f77b4"}
    name_palette = dict(zip(LABEL_NAMES, [palette[i] for i in range(3)]))

    # 图 1：三类分布
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([LABEL_NAMES[i] for i in counts.index],
                  counts.values, color=[palette[i] for i in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                ha="center", va="bottom", fontsize=12)
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution After Cleaning")
    plt.tight_layout()
    plt.savefig(FIG / "01_label_dist.png", dpi=150)
    plt.close()
    logger.info("01_label_dist.png 已存")

    # 图 2：source × label
    crosstab = pd.crosstab(df["source"], df["label"])
    crosstab.columns = [LABEL_NAMES[c] for c in crosstab.columns]
    fig, ax = plt.subplots(figsize=(7, 5))
    crosstab.plot(kind="bar", stacked=True, ax=ax,
                  color=[name_palette[n] for n in crosstab.columns])
    ax.set_xlabel("Source")
    ax.set_ylabel("Count")
    ax.set_title("Source × Label Distribution")
    ax.legend(title="Label")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG / "02_source_label.png", dpi=150)
    plt.close()
    logger.info("02_source_label.png 已存")

    # 图 3：标题长度（词数）直方图
    df["len_words"] = df["title"].astype(str).str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in [0, 1, 2]:
        ax.hist(df[df["label"] == c]["len_words"], bins=range(0, 40),
                alpha=0.55, label=LABEL_NAMES[c], color=palette[c])
    ax.set_xlabel("Title length (words)")
    ax.set_ylabel("Count")
    ax.set_title("Title Length Distribution by Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "03_title_length.png", dpi=150)
    plt.close()
    logger.info("03_title_length.png 已存")

    # 图 4：每类 Top-15 词
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for c in [0, 1, 2]:
        sub = df[df["label"] == c]["title"].astype(str).str.lower().tolist()
        cv = CountVectorizer(stop_words="english", token_pattern=r"[a-zA-Z]{3,}")
        X = cv.fit_transform(sub)
        freq = np.asarray(X.sum(axis=0)).ravel()
        words = np.array(cv.get_feature_names_out())
        idx = freq.argsort()[::-1][:15]
        axes[c].barh(words[idx][::-1], freq[idx][::-1], color=palette[c])
        axes[c].set_title(f"{LABEL_NAMES[c]}: Top-15 words")
        axes[c].tick_params(axis="y", labelsize=11)
    plt.tight_layout()
    plt.savefig(FIG / "04_top_words.png", dpi=150)
    plt.close()
    logger.info("04_top_words.png 已存")

    # 图 5：train/test 比例对照
    tr = train_df["label"].value_counts(normalize=True).sort_index()
    te = test_df["label"].value_counts(normalize=True).sort_index()
    cmp = pd.DataFrame({"train": tr.values, "test": te.values},
                       index=[LABEL_NAMES[i] for i in tr.index])
    fig, ax = plt.subplots(figsize=(7, 5))
    cmp.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Proportion")
    ax.set_title("Class Proportion: Train vs Test (Stratified Split)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG / "05_train_test.png", dpi=150)
    plt.close()
    logger.info("05_train_test.png 已存")

    # 图 6：每类长度箱线图
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [df[df["label"] == c]["len_words"].values for c in [0, 1, 2]]
    bp = ax.boxplot(data, tick_labels=LABEL_NAMES, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], [0, 1, 2]):
        patch.set_facecolor(palette[c])
        patch.set_alpha(0.7)
    ax.set_ylabel("Title length (words)")
    ax.set_title("Title Length Boxplot by Class")
    plt.tight_layout()
    plt.savefig(FIG / "06_length_box.png", dpi=150)
    plt.close()
    logger.info("06_length_box.png 已存")

    logger.info("数据可视化完成")


if __name__ == "__main__":
    main()
