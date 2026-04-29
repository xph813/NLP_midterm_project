"""
08 Qwen2.5-7B-Instruct 评测：三种 prompt × 500 条

模型从 ModelScope 下载（国内源），4bit 量化加载（仅 cuda 可用，
Mac 上自动降级为 fp16/bf16，会更慢但能跑）。
prompt：zero-shot / few-shot / CoT，每种跑完立即落盘，崩了能续。

测试集分层抽 500 条（每类至少 100 条），固定 random_state=42。
"""
from __future__ import annotations
import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (DATA_CLEAN, LABEL_NAMES, LOG_DIR, MODELS, OUT,
                   compute_metrics, get_device, plot_confusion, save_json,
                   save_predictions, set_seed, setup_hf_mirror, setup_logger)
setup_hf_mirror()

QWEN_DIR = OUT / "qwen"
SAMPLE_PATH = QWEN_DIR / "sample_500.tsv"


# ============= Prompt 模板 =============
SYSTEM_INSTRUCT = (
    "You are a news headline classifier. Classify the title into EXACTLY one of:\n"
    "- Hard:  hard news (politics, world events, business / finance, crime, "
    "policy, weather warnings, etc.).\n"
    "- Soft:  soft news (sports, entertainment / movies / TV / music / games, "
    "celebrity, food & drink, lifestyle, travel).\n"
    "- Other: everything else (health & medicine, automotive, kids content, "
    "general regional photos, etc.)."
)

FEW_SHOT_EXAMPLES = [
    ("Walmart Slashes Prices on Last-Generation iPads", "Hard"),
    ("Israeli sends observers to military drill in Morocco", "Hard"),
    ("Kim Kardashian shares rare photo of her children", "Soft"),
    ("How a favela in Rio got its clean water back, for $42300", "Hard"),
    ("50 Worst Habits For Belly Fat", "Other"),
    ("LeBron James scores 35 in Lakers comeback win", "Soft"),
]


def build_prompt(title: str, mode: str) -> list[dict]:
    """返回 chat-style messages，由 tokenizer.apply_chat_template 渲染。"""
    if mode == "zero":
        user = (f'Title: "{title}"\n'
                'Answer with only one of: Hard / Soft / Other.\n'
                'Format: "Answer: <label>"')
        return [{"role": "system", "content": SYSTEM_INSTRUCT},
                {"role": "user", "content": user}]
    if mode == "few":
        examples = "\n".join(
            [f'Title: "{t}"\nAnswer: {l}' for t, l in FEW_SHOT_EXAMPLES])
        user = (f"Here are some examples.\n{examples}\n\n"
                f'Now classify this:\nTitle: "{title}"\n'
                f'Format: "Answer: <label>"')
        return [{"role": "system", "content": SYSTEM_INSTRUCT},
                {"role": "user", "content": user}]
    if mode == "cot":
        user = (f'Title: "{title}"\n'
                "First, briefly explain your reasoning in ONE sentence, "
                'then output the final answer on a new line in the form '
                '"Final: <Hard/Soft/Other>".')
        return [{"role": "system", "content": SYSTEM_INSTRUCT},
                {"role": "user", "content": user}]
    raise ValueError(mode)


# ============= 答案抽取 =============
LABEL_TO_ID = {"hard": 0, "soft": 1, "other": 2}


def extract_label(text: str, mode: str) -> int:
    """从生成的文本中抽出标签 id；抽不到返回 -1。"""
    text = (text or "").strip()
    patterns = []
    if mode == "cot":
        patterns.append(r"final\s*[:：]\s*([A-Za-z]+)")
    patterns += [r"answer\s*[:：]\s*([A-Za-z]+)",
                 r"label\s*[:：]\s*([A-Za-z]+)",
                 r"\b(hard|soft|other)\b"]

    for p in patterns:
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            cand = m.group(1).lower()
            if cand in LABEL_TO_ID:
                return LABEL_TO_ID[cand]
            close = difflib.get_close_matches(cand, list(LABEL_TO_ID), n=1, cutoff=0.6)
            if close:
                return LABEL_TO_ID[close[0]]

    last = text.strip().splitlines()[-1] if text.strip() else ""
    close = difflib.get_close_matches(last.lower().strip(),
                                      list(LABEL_TO_ID), n=1, cutoff=0.5)
    if close:
        return LABEL_TO_ID[close[0]]
    return -1


# ============= 模型加载 =============
def load_qwen(model_id: str, logger):
    """优先用 modelscope 下；GPU 上 4bit，CPU/MPS 上 fp16/bf16。"""
    try:
        from modelscope import snapshot_download
        local_dir = snapshot_download(model_id, cache_dir=str(MODELS))
        logger.info(f"已就绪：{local_dir}")
    except Exception as e:
        logger.warning(f"modelscope 下载失败：{e}；改用 huggingface mirror")
        local_dir = model_id

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)

    device = get_device()
    if device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True,
                                     bnb_4bit_compute_dtype=torch.bfloat16,
                                     bnb_4bit_quant_type="nf4")
            model = AutoModelForCausalLM.from_pretrained(
                local_dir, quantization_config=bnb,
                device_map="auto", trust_remote_code=True)
            logger.info("已 4bit 量化加载")
        except Exception as e:
            logger.warning(f"4bit 加载失败 ({e})，回退到 bf16")
            model = AutoModelForCausalLM.from_pretrained(
                local_dir, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True)
    else:
        dtype = torch.float16 if device == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            local_dir, torch_dtype=dtype, trust_remote_code=True).to(device)
        logger.info(f"非 cuda 环境：以 {dtype} 加载到 {device}（会比较慢）")

    model.eval()
    return tokenizer, model


# ============= 推理主循环 =============
def run_mode(mode: str, sample_df: pd.DataFrame, tokenizer, model, logger,
             max_new_tokens: int, batch_size: int, force: bool):
    out_path = OUT / "predictions" / f"qwen_{mode}.tsv"
    if out_path.exists() and not force:
        logger.info(f"已存在 {out_path.name}，跳过")
        return pd.read_csv(out_path, sep="\t")

    import torch
    device = get_device()
    titles = sample_df["title"].astype(str).tolist()
    ids = sample_df["id"].tolist()
    y_true = sample_df["label"].astype(int).tolist()

    rows = []
    pbar = tqdm(range(0, len(titles), batch_size),
                desc=f"qwen-{mode}", ncols=90)
    for s in pbar:
        e = min(s + batch_size, len(titles))
        batch_titles = titles[s:e]
        prompts = [tokenizer.apply_chat_template(
            build_prompt(t, mode), tokenize=False,
            add_generation_prompt=True) for t in batch_titles]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = gen[:, inputs["input_ids"].shape[1]:]
        outs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for i, raw in enumerate(outs):
            pred = extract_label(raw, mode)
            rows.append({
                "id": ids[s + i],
                "title": batch_titles[i],
                "true_label": y_true[s + i],
                "pred_label": pred,
                "raw_response": raw.replace("\n", "\\n"),
            })
        if (s // batch_size) % 10 == 0:
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep="\t", index=False)
    logger.info(f"{mode}：拒答 / 抽不到 = {(df['pred_label'] == -1).sum()} 条")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n", type=int, default=500, help="抽样数")
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--modes", nargs="+", default=["zero", "few", "cot"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    QWEN_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ["tables", "metrics", "predictions", "figs"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    logger = setup_logger("qwen", LOG_DIR / "08_qwen_eval.log")

    if SAMPLE_PATH.exists() and not args.force:
        sample_df = pd.read_csv(SAMPLE_PATH, sep="\t")
        logger.info(f"复用已有抽样: {len(sample_df)} 条")
    else:
        test_df = pd.read_csv(DATA_CLEAN / "test.tsv", sep="\t")
        sample_df, _ = train_test_split(
            test_df, train_size=args.n, stratify=test_df["label"],
            random_state=args.seed)
        sample_df = sample_df.reset_index(drop=True)
        sample_df.to_csv(SAMPLE_PATH, sep="\t", index=False)
        logger.info(f"分层抽样 {len(sample_df)} 条 -> {SAMPLE_PATH}")

    for mode in args.modes:
        prompt_path = QWEN_DIR / "prompts" / f"{mode}.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(json.dumps(
            build_prompt("<TITLE>", mode), ensure_ascii=False, indent=2),
            encoding="utf-8")

    tokenizer, model = load_qwen(args.model, logger)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    summary = []
    for mode in args.modes:
        logger.info(f"\n===== Qwen prompt mode: {mode} =====")
        df = run_mode(mode, sample_df, tokenizer, model, logger,
                      args.max_new_tokens, args.bs, args.force)
        valid = df[df["pred_label"] != -1]
        y_t = valid["true_label"].astype(int).tolist()
        y_p = valid["pred_label"].astype(int).tolist()
        if not y_t:
            logger.warning(f"{mode} 全部抽取失败")
            continue
        m = compute_metrics(y_t, y_p)
        m["abstain_rate"] = float((df["pred_label"] == -1).mean())
        m["mode"] = mode
        save_json(m, OUT / "metrics" / f"qwen_{mode}.json")
        summary.append({"mode": mode, "accuracy": m["accuracy"],
                        "macro_f1": m["macro_f1"],
                        "weighted_f1": m["weighted_f1"],
                        "abstain": m["abstain_rate"]})
        logger.info(f"\n{m['report']}")
        plot_confusion(
            y_t, y_p, OUT / "figs" / f"10_qwen_{mode}_confusion.png",
            title=f"Qwen ({mode}) – Confusion Matrix")

    if summary:
        sdf = pd.DataFrame(summary)
        sdf.to_csv(OUT / "tables" / "qwen_prompt_compare.csv", index=False)
        logger.info(f"\nPrompt 对比:\n{sdf.to_string(index=False)}")

        best = sdf.sort_values("macro_f1", ascending=False).iloc[0]
        best_mode = best["mode"]
        logger.info(f"最佳 prompt: {best_mode}")
        src = OUT / "predictions" / f"qwen_{best_mode}.tsv"
        dst = OUT / "predictions" / "qwen_test.tsv"
        if src.exists():
            import shutil
            shutil.copyfile(src, dst)
        save_json({"mode": best_mode, **best.to_dict()},
                  OUT / "metrics" / "qwen_test.json")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(sdf))
        w = 0.35
        ax.bar(x - w / 2, sdf["accuracy"], w, label="accuracy")
        ax.bar(x + w / 2, sdf["macro_f1"], w, label="macro F1")
        ax.set_xticks(x)
        ax.set_xticklabels(sdf["mode"])
        ax.set_ylim(0, 1)
        ax.set_title("Qwen prompt strategy comparison")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT / "figs" / "11_qwen_prompt_compare.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
