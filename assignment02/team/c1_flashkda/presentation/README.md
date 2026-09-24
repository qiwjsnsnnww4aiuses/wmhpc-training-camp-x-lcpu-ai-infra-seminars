# C1 FlashKDA LaTeX 汇报

## 文件

- `C1_FlashKDA_汇报.tex`：16:9 Beamer 源码，18 页（含标题页和 Q&A）。
- 编译后得到 `C1_FlashKDA_汇报.pdf`。

内容按 10～12 分钟设计，源码中的 `\note{...}` 是逐页讲稿提示，默认不会显示在投影 PDF 中。

## 编译

需要 XeLaTeX 和 `ctex`、`beamer`、`tikz`、`booktabs`。

```bash
cd /home/wpy/documents/lcpu2026/assignment02/team/c1_flashkda/presentation
xelatex -interaction=nonstopmode -halt-on-error C1_FlashKDA_汇报.tex
xelatex -interaction=nonstopmode -halt-on-error C1_FlashKDA_汇报.tex
```

也可使用 `latexmk`：

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error C1_FlashKDA_汇报.tex
```

## 导出 PowerPoint

本作业要求 LaTeX 汇报时，直接使用 PDF 放映最稳。如果必须提交 `.pptx`，可用 LibreOffice Impress、PowerPoint 或 Adobe Acrobat 将 PDF 每页导入为一张幻灯片；但转换后的公式和图形通常不可编辑。

## 演讲节奏

```text
1--4 页    问题、KDA 依赖、项目调用链       约 2.5 分钟
5--10 页   复现、CHUNK、tile、NCU、并行度   约 4.0 分钟
11--14 页  挑战、正确性、性能、慢因         约 3.0 分钟
15--17 页  精度方案、发布决策、结论           约 2.0 分钟
```

若现场只有 8 分钟，可跳过第 8 页 microbench 细节和第 10 页候选方案表，只口头报结论。

## 讲稿备注的显示方法（可选）

如需生成带备注的讲者版本，在导言区加入：

```latex
\usepackage{pgfpages}
\setbeameroption{show notes on second screen=right}
```

普通投影版不要加入这两行。

