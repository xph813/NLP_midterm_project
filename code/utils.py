"""共用工具：路径常量、日志、随机种子、设备探测、metrics、画图、predictions 落盘。
所有脚本都从这里 import。
"""
from __future__ import annotations
import os
import json
import logging
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _setup_matplotlib_fonts():
    """所有图统一用 Times New Roman 风格（serif）。
    Linux 默认没装 Times New Roman，按优先级回退到 Liberation Serif / DejaVu Serif。
    可装 sudo apt install ttf-mscorefonts-installer 后重启 python，会自动用真 TNR。
    """
    try:
        import matplotlib
        from matplotlib import font_manager
        candidates = ["Times New Roman", "Times", "Liberation Serif",
                      "Nimbus Roman", "DejaVu Serif", "FreeSerif"]
        avail = {f.name for f in font_manager.fontManager.ttflist}
        fonts = [c for c in candidates if c in avail] or ["serif"]
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = fonts
        matplotlib.rcParams["mathtext.fontset"] = "stix"
        matplotlib.rcParams["axes.unicode_minus"] = False
    except ImportError:
        pass


_setup_matplotlib_fonts()


ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "数据集"
DATA_CLEAN = ROOT / "data_clean"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
LOG_DIR = OUT / "logs"

LABELS = [0, 1, 2]
LABEL_NAMES = ["Hard", "Soft", "Other"]
LABEL_NAMES_CN = ["硬新闻", "软新闻", "其他"]
NUM_LABELS = 3


def setup_logger(name: str = "run", log_file: Optional[Path] = None,
                 level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device() -> str:
    """3090 上返回 cuda，Mac M1 返回 mps，其他返回 cpu。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def setup_hf_mirror() -> None:
    """国内 HuggingFace 镜像，避免下载慢。"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# ===== 模型源映射：hf 名 -> modelscope 名 =====
HF_TO_MODELSCOPE = {
    "bert-base-uncased": "AI-ModelScope/bert-base-uncased",
    "roberta-base": "AI-ModelScope/roberta-base",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
}


def get_model_path(name: str, logger=None) -> str:
    """统一模型解析。优先级：
       1. 本地 huggingface cache 命中 -> 用 hf 名（from_pretrained 自动读本地）
       2. modelscope 下载到 models/（已有则跳过）
       3. 都失败 -> 回退原 hf 名
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        fp = try_to_load_from_cache(repo_id=name, filename="config.json")
        if fp and Path(fp).exists():
            if logger:
                logger.info(f"模型 {name} hf cache 命中，用本地缓存（不重下）")
            return name
    except Exception:
        pass

    ms_id = HF_TO_MODELSCOPE.get(name, name)
    try:
        from modelscope import snapshot_download
        path = snapshot_download(ms_id, cache_dir=str(MODELS))
        if logger:
            logger.info(f"模型 {name} 就绪 (modelscope) -> {path}")
        return path
    except Exception as e:
        if logger:
            logger.warning(f"modelscope 下载 {ms_id} 失败 ({e})；回退到 hf 名 {name}")
        return name


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(y_true, y_pred, labels: Optional[Sequence[int]] = None) -> dict:
    """精度/召回/F1，包括 macro / weighted / per-class。"""
    from sklearn.metrics import (accuracy_score,
                                 precision_recall_fscore_support,
                                 classification_report)
    if labels is None:
        labels = LABELS
    acc = accuracy_score(y_true, y_pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0, labels=labels)
    p_pc, r_pc, f_pc, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=labels)
    out = {
        "accuracy": float(acc),
        "macro_precision": float(p_m),
        "macro_recall": float(r_m),
        "macro_f1": float(f_m),
        "weighted_precision": float(p_w),
        "weighted_recall": float(r_w),
        "weighted_f1": float(f_w),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": float(p_pc[i]),
                "recall": float(r_pc[i]),
                "f1": float(f_pc[i]),
                "support": int(sup[i]),
            } for i in range(len(labels))
        },
        "report": classification_report(
            y_true, y_pred, target_names=LABEL_NAMES,
            digits=4, zero_division=0)
    }
    return out


def plot_confusion(y_true, y_pred, save_path, title: str = "Confusion Matrix"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_predictions(ids, titles, y_true, y_pred, save_path,
                     probs: Optional[np.ndarray] = None) -> None:
    """每条样本一行，方便后面集成 / 错例分析。"""
    import pandas as pd
    data = {"id": ids, "title": titles, "true_label": y_true,
            "pred_label": y_pred}
    df = pd.DataFrame(data)
    if probs is not None:
        for i, name in enumerate(LABEL_NAMES):
            df[f"prob_{name}"] = probs[:, i]
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, sep="\t", index=False)
