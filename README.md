# NLP 期中作业：MIND + GOOGLE 新闻标题三分类

本仓库完成自然语言处理课程期中任务：依据 `MIND和GOOGLE数据分类系统.xlsx` 将英文新闻标题划分为 **硬新闻 / 软新闻 / 其他** 三类，分别使用 **FastText、BERT/RoBERTa 微调、Qwen2.5 提示推理** 在同一测试集上进行对比。

- 代码：<https://github.com/xph813/NLP_midterm_project.git>
- 报告：见 `report/main.pdf`（源文件 `report/main.tex`）

## 实验结论（摘要）

BERT-base 微调在 macro-F1 上表现最好；FastText 训练与推理成本低、效果仍可用于基线；Qwen-7B 在本任务（标签由 Excel 映射规则定义）上明显落后于微调模型，详细分析见报告。

## 环境

建议在 **CPU 环境** 完成数据清洗与 FastText；在 **GPU（如 RTX 3090）** 上完成 BERT 与 Qwen 实验。

```bash
conda create -n nlp python=3.10 -y
conda activate nlp
pip install -r requirements.txt

# 若已有含 PyTorch 的环境，可只补其余依赖：
conda activate <你的环境名>
pip install -r requirements_no_torch.txt

# 完整 GPU 依赖（含 PyTorch）见：
pip install -r requirements_gpu.txt
```

国内访问 HuggingFace 可设置镜像（代码中亦有设置）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Qwen 等模型可通过 **ModelScope** 下载：`pip install modelscope`。

## 运行顺序

按脚本编号 `01`～`10` 依次执行即可；`11`～`14` 为消融与扩展实验，可选。各脚本支持 **断点续跑**（已有输出会跳过）；需完全重跑时加 `--force`。

| # | 脚本 | 说明 | 建议环境 | 约耗时 |
|---|---|---|---|---|
| 01 | `01_clean_data.py` | 读 Excel、清洗、9:1 分层划分 | CPU | &lt; 1 min |
| 02 | `02_data_viz.py` | 6 张数据可视化 | CPU | &lt; 1 min |
| 03 | `03_fasttext_grid.py` | FastText 网格搜索（18 组） | CPU | ~30 min |
| 04 | `04_fasttext_aug.py` | 数据增强与 test 评测 | CPU | ~10 min |
| 05 | `05_bert_train.py` | bert-base、roberta-base 训练 | GPU | ~20 min |
| 06 | `06_bert_ablation.py` | max_len 消融 | GPU | ~15 min |
| 07 | `07_bert_eval.py` | BERT 在 test 上评测 | GPU | ~2 min |
| 08 | `08_qwen_eval.py` | Qwen 三种 prompt，500 条抽样 | GPU | ~40 min |
| 09 | `09_analysis.py --phase 1` | 指标汇总、错例导出 | CPU/GPU | &lt; 5 min |
| — | 人工标注错因 | 填好 `error_analysis_labeled.csv` 后 | — | — |
| 09 | `09_analysis.py --phase 2` | 错因饼图 | CPU | &lt; 1 min |
| 10 | `10_ensemble.py` | 三模型多数投票 | CPU | &lt; 1 min |

可选脚本：`11_bert_lr_ablation.py`、`12_bert_learning_curve.py`、`13_qwen_size_compare.py`、`14_qwen_fewshot_count.py`。

## 目录说明

`models/`、`outputs/`、`data_clean/` 体积较大，已写入 `.gitignore`；克隆本仓库后需自行运行脚本生成。原始数据位于 `数据集/`。

```
数据集/          原始 tsv 与 Excel
data_clean/      清洗与划分结果（脚本生成）
models/          预训练权重（勿提交）
outputs/         fasttext、bert、qwen、predictions、metrics、tables、figs、logs 等
report/          LaTeX 报告与插图
```

## 说明

BERT 训练初期 loss 可能先升后降，与分类头随机初始化有关，属常见现象。
