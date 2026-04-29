# 编译与使用说明

## 目录结构

```
report/
├── main.tex              主文件（论文正文）
├── reference.bib         参考文献
├── SYSUReport.cls        SYSU 模板格式控制（不要改）
├── figures/              图片（已自动从 outputs/figs/ 复制 25 张实验图）
├── main.pdf              当前编译结果（28 页）
├── README.md             SYSU 模板原始 README
└── HOWTO.md              本文件
```

## 编译步骤

模板要求 **xelatex**（中文支持），编译顺序固定：

```bash
cd report
xelatex main.tex     # 第 1 次：生成 .aux
bibtex main          # 处理参考文献
xelatex main.tex     # 第 2 次：解析交叉引用、bib
xelatex main.tex     # 第 3 次：表格 / 图序号稳定
```

或者一行干完：

```bash
xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex
```

如果用 TeXstudio / TeXworks，把构建器选成 `xelatex+bibtex+xelatex*2` 即可。
推荐 Overleaf 在线编辑：直接把整个 `report/` 文件夹打 zip 上传，
项目设置里把编译器改成 **XeLaTeX** 即可。

## 必须填的内容

打开 `main.tex` 顶部，把空着的 4 个字段填上：

```latex
\stuname{你的姓名}
\stuid{你的学号}
\inst{你所在学院}
\major{你所在专业}
```

填完之后封面会自动渲染好。

## 主要章节速览

| 章节 | 页数 | 主要图/表 |
|---|---|---|
| 1 任务背景与数据 | 4–6 | 表 1–2 标签体系 |
| 2 算法原理 | 6–8 | 公式 1–4 |
| 3 数据清洗与可视化 | 8–10 | 图 1–2（6 张数据图）、表 3 |
| 4 FastText | 10–12 | 表 4–6、图 3 |
| 5 BERT | 12–17 | 表 7–11、图 4–7 |
| 6 Qwen2.5 | 17–20 | 表 12–14、图 8–11 |
| 7 综合实验分析 | 20–24 | 表 15–18、图 12–14 |
| 8 总结与场景适用 | 24–26 | 表 19 |
| 参考文献 | 27 | 8 篇 |

## 修改图片

如果以后实验数据更新了，重新生成 figures：

```bash
cd /home/xph/Desktop/NLP_midterm_project
cp outputs/figs/*.png report/figures/
cd report && xelatex main && bibtex main && xelatex main && xelatex main
```

## 已知小问题

剩 4 个 Overfull \\hbox 警告（最大 22pt，肉眼几乎不可见，是中英文混排
+ 长 \\texttt 段落导致的。LaTeX 输出 PDF 仍然完整，可以忽略。
