"""
10 三模型多数投票集成

在 Qwen 评测的同 500 条样本上，FastText + BERT + Qwen 多数投票：
- 三票相同：取该标签
- 三票不同：以概率最高那个为准（用 BERT 概率，否则 FastText 概率，否则 Qwen 投硬票）
看集成后的 macro_f1 能否超过单模型最佳。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (LABEL_NAMES, LOG_DIR, OUT, compute_metrics, plot_confusion,
                   save_json, save_predictions, setup_logger)

PRED = OUT / "predictions"


def main():
    for sub in ["tables", "metrics", "predictions", "figs"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    logger = setup_logger("ensemble", LOG_DIR / "10_ensemble.log")
    ft = pd.read_csv(PRED / "fasttext_test.tsv", sep="\t")
    bert = pd.read_csv(PRED / "bert_test.tsv", sep="\t")
    qwen = pd.read_csv(PRED / "qwen_test.tsv", sep="\t")
    qwen = qwen[qwen["pred_label"] != -1]

    common_ids = (set(ft["id"].astype(str))
                  & set(bert["id"].astype(str))
                  & set(qwen["id"].astype(str)))

    def _filter_and_index(df):
        df = df.copy()
        df["_id_str"] = df["id"].astype(str)
        df = df[df["_id_str"].isin(common_ids)].set_index("_id_str")
        return df

    ft = _filter_and_index(ft)
    bert = _filter_and_index(bert)
    qwen = _filter_and_index(qwen)

    ids = sorted(common_ids)
    ft = ft.loc[ids]; bert = bert.loc[ids]; qwen = qwen.loc[ids]

    rows = []
    for rid in ids:
        ft_p = int(ft.loc[rid, "pred_label"])
        bert_p = int(bert.loc[rid, "pred_label"])
        qwen_p = int(qwen.loc[rid, "pred_label"])
        true = int(qwen.loc[rid, "true_label"])
        votes = [ft_p, bert_p, qwen_p]
        from collections import Counter
        cnt = Counter(votes)
        top, n = cnt.most_common(1)[0]
        if n >= 2:
            ens = top
        else:
            ens = bert_p
        rows.append({"id": rid, "title": qwen.loc[rid, "title"],
                     "true_label": true, "pred_label": ens,
                     "ft": ft_p, "bert": bert_p, "qwen": qwen_p})

    out = pd.DataFrame(rows)
    out.to_csv(PRED / "ensemble_test.tsv", sep="\t", index=False)

    y_t = out["true_label"].astype(int).tolist()
    y_p = out["pred_label"].astype(int).tolist()
    m = compute_metrics(y_t, y_p)
    save_json(m, OUT / "metrics" / "ensemble.json")
    logger.info(f"\n{m['report']}")
    logger.info(f"集成 accuracy={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}")

    plot_confusion(y_t, y_p, OUT / "figs" / "16_ensemble_confusion.png",
                   title="Majority Vote Ensemble – Confusion Matrix")

    rows = [{"model": k, "accuracy": v["accuracy"],
             "macro_f1": v["macro_f1"], "weighted_f1": v["weighted_f1"]}
            for k, v in [("FastText",
                          compute_metrics(out["true_label"], out["ft"])),
                         ("BERT",
                          compute_metrics(out["true_label"], out["bert"])),
                         ("Qwen",
                          compute_metrics(out["true_label"], out["qwen"])),
                         ("Ensemble", m)]]
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(OUT / "tables" / "ensemble_compare.csv", index=False)
    logger.info(f"\n{cmp_df.round(4).to_string(index=False)}")


if __name__ == "__main__":
    main()
