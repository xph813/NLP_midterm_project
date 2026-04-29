"""
00 统一下载所有要用的预训练模型（modelscope 国内源）。

可以提前一次性跑完，然后离线训练。
已下载过的会自动跳过（modelscope 自带去重）。

用法:
    python 00_download_models.py              # 全部下
    python 00_download_models.py --only roberta qwen-1.5b
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import LOG_DIR, MODELS, get_model_path, setup_logger


CATALOG = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+",
                        choices=list(CATALOG.keys()) + ["all"],
                        default=["all"],
                        help="想下载哪些（不写就全下）")
    args = parser.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("download", LOG_DIR / "00_download.log")
    keys = list(CATALOG.keys()) if "all" in args.only else args.only

    logger.info(f"将下载: {keys}")
    for k in keys:
        hf_name = CATALOG[k]
        logger.info(f"\n========= 下载 {k}  ({hf_name}) =========")
        t0 = time.time()
        path = get_model_path(hf_name, logger=logger)
        logger.info(f"  耗时 {time.time() - t0:.0f} 秒；路径 = {path}")


if __name__ == "__main__":
    main()
