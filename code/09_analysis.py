"""
09 综合分析（最终报告的"实验结果"章节素材都从这里出）

phase 1（默认跑这个）：
  - 三模型指标合并大表 outputs/tables/final_metrics.csv
  - 三模型每类 F1 分组柱状图
  - 三混淆矩阵并排 subplot
  - 三模型错误重叠维恩图（matplotlib_venn）
  - 推理速度 / 成本对比表（手填，给个模板 csv）
  - 抽 30 条三模型不一致的样本，dump 到 error_analysis_to_label.csv

phase 2（人工标完错因后）：
  python 09_analysis.py --phase 2
  从 error_analysis_labeled.csv 读 "error_type" 列，画错因饼图。

注意：Qwen 只跑了 500 条抽样，跟另外两个模型 test 全集对不上。
所以三模型对比时，**自动把 FastText/BERT 的预测限制到 Qwen 的同 500 条样本**，
保证可比；同时报告里也会单独列 FastText/BERT 在全 test 上的指标。
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (DATA_CLEAN, LABEL_NAMES, LOG_DIR, OUT, compute_metrics,
                   load_json, plot_confusion, save_json, set_seed,
                   setup_logger)

PRED_DIR = OUT / "predictions"
METRIC_DIR = OUT / "metrics"
FIG = OUT / "figs"
TAB = OUT / "tables"


def load_predictions(name: str) -> pd.DataFrame:
    path = PRED_DIR / f"{name}_test.tsv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    df["model"] = name
    return df


def phase1(args, logger):
    ft = load_predictions("fasttext")
    bert = load_predictions("bert")
    qwen = load_predictions("qwen")
    if ft is None or bert is None or qwen is None:
        logger.error("缺预测文件：请先把 04 / 07 / 08 跑完")
        return

    qwen_ids = set(qwen["id"].astype(str).tolist())
    ft_sub = ft[ft["id"].astype(str).isin(qwen_ids)].reset_index(drop=True)
    bert_sub = bert[bert["id"].astype(str).isin(qwen_ids)].reset_index(drop=True)
    qwen_v = qwen[qwen["pred_label"] != -1].reset_index(drop=True)
    ft_sub = ft_sub[ft_sub["id"].astype(str).isin(qwen_v["id"].astype(str))]
    bert_sub = bert_sub[bert_sub["id"].astype(str).isin(qwen_v["id"].astype(str))]

    rows = []
    for name, full, sub in [
        ("FastText (full test)", ft, None),
        ("BERT (full test)", bert, None),
        ("FastText (sample 500)", None, ft_sub),
        ("BERT (sample 500)", None, bert_sub),
        ("Qwen (sample 500)", None, qwen_v),
    ]:
        df = full if full is not None else sub
        m = compute_metrics(df["true_label"].astype(int).tolist(),
                            df["pred_label"].astype(int).tolist())
        rows.append({
            "model": name, "n": len(df),
            "accuracy": m["accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            **{f"f1_{LABEL_NAMES[i]}": m["per_class"][LABEL_NAMES[i]]["f1"]
               for i in range(3)},
        })
    metric_df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    metric_df.to_csv(TAB / "final_metrics.csv", index=False)
    logger.info(f"\n{metric_df.round(4).to_string(index=False)}")

    import matplotlib.pyplot as plt
    sub_models = [("FastText", ft_sub), ("BERT", bert_sub), ("Qwen", qwen_v)]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3); w = 0.25
    for i, (name, df) in enumerate(sub_models):
        m = compute_metrics(df["true_label"].astype(int).tolist(),
                            df["pred_label"].astype(int).tolist())
        f1s = [m["per_class"][LABEL_NAMES[c]]["f1"] for c in range(3)]
        ax.bar(x + (i - 1) * w, f1s, w, label=name)
    ax.set_xticks(x); ax.set_xticklabels(LABEL_NAMES)
    ax.set_ylim(0, 1); ax.set_ylabel("F1")
    ax.set_title("Per-class F1 across models (same 500 samples)")
    ax.legend(); plt.tight_layout()
    plt.savefig(FIG / "12_per_class_f1.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    for ax, (name, df) in zip(axes, sub_models):
        cm = confusion_matrix(df["true_label"].astype(int),
                              df["pred_label"].astype(int), labels=[0, 1, 2])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
        ax.set_title(name); ax.set_xlabel("Pred"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(FIG / "13_confusion_all.png", dpi=150); plt.close()

    err = {}
    for name, df in sub_models:
        df = df.copy()
        df["wrong"] = df["true_label"].astype(int) != df["pred_label"].astype(int)
        err[name] = set(df[df["wrong"]]["id"].astype(str).tolist())
    try:
        from matplotlib_venn import venn3
        fig, ax = plt.subplots(figsize=(7, 6))
        venn3([err["FastText"], err["BERT"], err["Qwen"]],
              ("FastText", "BERT", "Qwen"))
        ax.set_title("Error overlap on the same 500 samples")
        plt.tight_layout()
        plt.savefig(FIG / "14_error_venn.png", dpi=150); plt.close()
    except ImportError:
        logger.warning("matplotlib_venn 未安装，跳过维恩图")

    cost_path = TAB / "cost_compare.csv"
    if not cost_path.exists():
        cost_df = pd.DataFrame([
            {"model": "FastText", "params": "~1M", "train_time_min": "",
             "inference_ms_per_sample": "", "memory_mb": ""},
            {"model": "BERT-base", "params": "110M", "train_time_min": "",
             "inference_ms_per_sample": "", "memory_mb": ""},
            {"model": "RoBERTa-base", "params": "125M", "train_time_min": "",
             "inference_ms_per_sample": "", "memory_mb": ""},
            {"model": "Qwen2.5-7B-4bit", "params": "7B (4bit)",
             "train_time_min": "0 (no train)",
             "inference_ms_per_sample": "", "memory_mb": ""},
        ])
        cost_df.to_csv(cost_path, index=False)
        logger.info(f"已生成成本对比模板 {cost_path}（人工填数）")

    rows = []
    for _, row in qwen_v.iterrows():
        rid = str(row["id"])
        ft_p = ft_sub[ft_sub["id"].astype(str) == rid]["pred_label"]
        bert_p = bert_sub[bert_sub["id"].astype(str) == rid]["pred_label"]
        if ft_p.empty or bert_p.empty:
            continue
        ft_p = int(ft_p.iloc[0]); bert_p = int(bert_p.iloc[0])
        qwen_p = int(row["pred_label"])
        true = int(row["true_label"])
        if len({ft_p, bert_p, qwen_p}) > 1:
            rows.append({
                "id": rid, "title": row["title"], "true_label": true,
                "true_name": LABEL_NAMES[true],
                "fasttext_pred": LABEL_NAMES[ft_p],
                "bert_pred": LABEL_NAMES[bert_p],
                "qwen_pred": LABEL_NAMES[qwen_p],
                "error_type": ""
            })
    err_df = pd.DataFrame(rows)
    err_df = err_df.sample(min(30, len(err_df)), random_state=42).reset_index(drop=True)
    err_df.to_csv(OUT / "error_analysis_to_label.csv", index=False)
    logger.info(f"已 dump {len(err_df)} 条不一致样本到 error_analysis_to_label.csv，"
                "请人工填 error_type 列后跑 --phase 2")


def phase2(args, logger):
    path = OUT / "error_analysis_labeled.csv"
    if not path.exists():
        logger.error(f"找不到 {path}；请把 to_label.csv 标好 error_type 后改名")
        return
    import matplotlib.pyplot as plt
    df = pd.read_csv(path)
    cnt = df["error_type"].fillna("(empty)").value_counts()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(cnt.values, labels=cnt.index, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Error type distribution (N={len(df)})")
    plt.tight_layout()
    plt.savefig(FIG / "15_error_pie.png", dpi=150); plt.close()
    logger.info(f"15_error_pie.png 已存")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1)
    args = parser.parse_args()
    set_seed(42)
    logger = setup_logger("analysis", LOG_DIR / "09_analysis.log")
    if args.phase == 1:
        phase1(args, logger)
    else:
        phase2(args, logger)


if __name__ == "__main__":
    main()
