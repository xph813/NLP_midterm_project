"""
01 数据清洗 + 9:1 分层划分

读 Excel 构建 (主类, 子类) -> {0,1,2} 的映射，按映射给两个 tsv 打标签，
丢掉 video 整大类、D 列空、标题空、重复标题，最后按标签分层 9:1 划分。

输出：
  data_clean/labels_map.json
  data_clean/clean.tsv
  data_clean/train.tsv
  data_clean/test.tsv
  outputs/logs/01_clean.log
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import openpyxl
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (DATA_RAW, DATA_CLEAN, LOG_DIR, save_json, set_seed,
                   setup_logger)


def build_label_map(excel_path: Path, logger):
    """解析 Excel：A 是主类标题行（B/C/D 通常空），B 非空时一行就是 (主类, 子类) -> label。
    若主类那行 D 列写了"该分类整体不使用"（如 video），则该主类下所有子类全部跳过。"""
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    label_map: dict[tuple[str, str], int] = {}
    skipped_main = []

    for sheet_name in ["MIND", "GOOGLE"]:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        current_main = None
        skip_main = False

        for row in rows[1:]:
            a = str(row[0]).strip() if row[0] is not None else ""
            b = str(row[1]).strip() if row[1] is not None else ""
            d_raw = row[3]

            if a:
                current_main = a
                skip_main = False
                if d_raw is not None and "不使用" in str(d_raw):
                    skip_main = True
                    skipped_main.append(f"{sheet_name}:{current_main}")
                continue

            if not b or skip_main or current_main is None:
                continue

            try:
                label = int(d_raw)
            except (TypeError, ValueError):
                continue
            if label not in (0, 1, 2):
                continue

            label_map[(current_main, b)] = label

    logger.info(f"标签映射条目数: {len(label_map)}")
    logger.info(f"被丢弃的主分类: {skipped_main}")
    return label_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="即使产物已存在也重新生成")
    args = parser.parse_args()

    set_seed(args.seed)
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("clean", LOG_DIR / "01_clean.log")

    train_path = DATA_CLEAN / "train.tsv"
    test_path = DATA_CLEAN / "test.tsv"
    if train_path.exists() and test_path.exists() and not args.force:
        logger.info("train/test 已存在，跳过；--force 强制重跑")
        return

    excel_path = DATA_RAW / "MIND和GOOGLE数据分类系统.xlsx"
    label_map = build_label_map(excel_path, logger)
    save_json({f"{k[0]}|||{k[1]}": v for k, v in label_map.items()},
              DATA_CLEAN / "labels_map.json")

    cols = ["id", "main_cat", "sub_cat", "title"]
    mind = pd.read_csv(DATA_RAW / "MIND.tsv", sep="\t", header=None,
                       names=cols, on_bad_lines="skip", quoting=3,
                       dtype=str, keep_default_na=False)
    google = pd.read_csv(DATA_RAW / "GOOGLE.tsv", sep="\t", header=None,
                         names=cols, on_bad_lines="skip", quoting=3,
                         dtype=str, keep_default_na=False)
    mind["source"] = "MIND"
    google["source"] = "GOOGLE"
    df = pd.concat([mind, google], ignore_index=True)
    n0 = len(df)
    logger.info(f"原始合计: {n0} (MIND={len(mind)}, GOOGLE={len(google)})")

    df = df[(df["title"].str.strip() != "") &
            (df["main_cat"].str.strip() != "") &
            (df["sub_cat"].str.strip() != "")]
    n1 = len(df)
    logger.info(f"去除空字段后: {n1} (-{n0 - n1})")

    df["label"] = df.apply(
        lambda r: label_map.get((r["main_cat"], r["sub_cat"]), -1), axis=1)
    df = df[df["label"] != -1]
    n2 = len(df)
    logger.info(f"按 Excel 映射打标后: {n2} (-{n1 - n2})")

    df["title"] = (df["title"].astype(str).str.strip()
                   .str.replace(r"\s+", " ", regex=True))
    df = df[df["title"].str.len() > 0]
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    n3 = len(df)
    logger.info(f"去重并清理空白后: {n3} (-{n2 - n3})")

    cnt = df["label"].value_counts().sort_index().to_dict()
    logger.info(f"清洗后类别分布: {cnt}")
    df[["id", "source", "main_cat", "sub_cat", "title", "label"]].to_csv(
        DATA_CLEAN / "clean.tsv", sep="\t", index=False)

    train_df, test_df = train_test_split(
        df, test_size=0.1, stratify=df["label"], random_state=args.seed)
    train_df.to_csv(train_path, sep="\t", index=False)
    test_df.to_csv(test_path, sep="\t", index=False)
    logger.info(f"训练集: {len(train_df)}  测试集: {len(test_df)}")
    logger.info(f"训练集分布: {train_df['label'].value_counts().sort_index().to_dict()}")
    logger.info(f"测试集分布: {test_df['label'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
