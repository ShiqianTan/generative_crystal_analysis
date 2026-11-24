# FieldForge / Crystalite crystal benchmark analysis

这是一个可直接作为 GitHub 仓库使用的、数据与代码分离的分析目录。默认比较
`FieldForge (PT)` 与 `Crystalite`，不包含原始 CIF、MLIP JSON、MP20 数据、虚拟环境或
历史生成图片。

## 分析内容

- 空间群、Wyckoff multiplicity 和 Wyckoff dimensionality；
- 元素频率、周期表热图和每个材料的独立元素数；
- 相对 MP20 的密度与元素数 Wasserstein 距离；
- 多 MLIP 的能量、弛豫 RMSD、稳定/亚稳比例及异常值；
- MP20 合并对比图；
- 按空间群选择、复制并渲染代表性结构。

每张 PNG 图会同时生成 PDF，便于论文排版。

## 目录结构

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── scripts/
│   ├── settings.py
│   ├── run_all.py
│   ├── plot_crystal_symmetry_distributions.py
│   ├── plot_element_distributions.py
│   ├── plot_wasserstein_similarity.py
│   ├── analyze_mlip_relaxation_metrics.py
│   ├── add_mp20_reference_outputs.py
│   ├── select_representative_structures.py
│   ├── update_fieldforge_pt_elements.py
│   └── update_fieldforge_pt_metrics.py
└── tests/
```

`data/` 与 `outputs/` 均被 `.gitignore` 排除。代码默认识别以下本地数据布局：

```text
data/
├── cifs/
│   ├── fieldforge/*.cif
│   └── crystalite/*.cif
├── mlip_json/*.json
└── reference/mp20_total.csv
```

数据也可以放在任意位置，通过命令行参数或环境变量传入，无需修改源码。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Python 3.10 及以上版本受支持。

## 一键运行

只分析两个 CIF 集合：

```bash
python scripts/run_all.py \
  --fieldforge-cifs /path/to/fieldforge/cifs \
  --crystalite-cifs /path/to/crystalite/cifs \
  --skip wasserstein \
  --skip mp20 \
  --skip mlip
```

运行完整流程：

```bash
python scripts/run_all.py \
  --fieldforge-cifs /path/to/fieldforge/cifs \
  --crystalite-cifs /path/to/crystalite/cifs \
  --mp20-csv /path/to/mp20_total.csv \
  --mlip-json-dir /path/to/mlip_json \
  --output-dir outputs \
  --limit 2500
```

流水线按以下顺序运行：symmetry → elements → Wasserstein → MP20 → MLIP。没有提供
MP20 或 MLIP 路径时，相应阶段会自动跳过。使用 `--skip STAGE` 可主动跳过任意阶段。

## 分步运行

所有脚本都支持 `--help`，且模型路径可以在命令行显式传入。

### 1. 晶体对称性

```bash
python scripts/plot_crystal_symmetry_distributions.py \
  --model 'FieldForge (PT)' /path/to/fieldforge/cifs \
  --model Crystalite /path/to/crystalite/cifs \
  --limit 2500 \
  --spacegroup-comparison-output outputs/symmetry/figures/spacegroup_horizontal_top15.png \
  --individual-spacegroup-dir outputs/symmetry/figures/per_model
```

默认使用 `symprec=0.1`、`angle_tolerance=5.0`。加 `--use-cache` 可只重画图片而不重新解析 CIF。

### 2. 元素分布

```bash
python scripts/plot_element_distributions.py \
  --model 'FieldForge (PT)' /path/to/fieldforge/cifs \
  --model Crystalite /path/to/crystalite/cifs \
  --limit 2500
```

### 3. MP20 Wasserstein 距离

```bash
FIELD_FORGE_CIF_DIR=/path/to/fieldforge/cifs \
CRYSTALITE_CIF_DIR=/path/to/crystalite/cifs \
python scripts/plot_wasserstein_similarity.py \
  --mp20-csv /path/to/mp20_total.csv
```

输出包括晶体密度的 `W_rho` 和每个结构独立元素数的 `W_N_el`。距离越小，分布越接近 MP20。

### 4. MP20 合并图

先完成 symmetry 和 elements，再执行：

```bash
python scripts/add_mp20_reference_outputs.py \
  --mp20-csv /path/to/mp20_total.csv
```

### 5. 多 MLIP 指标

```bash
python scripts/analyze_mlip_relaxation_metrics.py \
  --base-dir /path/to/mlip_json \
  --element-cache outputs/elements/csv/structure_element_cache.csv \
  --mp20-csv /path/to/mp20_total.csv
```

默认 JSON glob 在 `scripts/settings.py` 中集中定义。若文件命名不同，可重复使用：

```bash
python scripts/analyze_mlip_relaxation_metrics.py \
  --base-dir /path/to/mlip_json \
  --model 'FieldForge (PT)' 'fieldforge-*.json' /path/to/fieldforge/cifs \
  --model Crystalite 'crystalite-*.json' /path/to/crystalite/cifs
```

### 6. 代表性结构

该步骤需要合并后的 MP20 symmetry CSV 以及 MLIP JSON：

```bash
python scripts/select_representative_structures.py \
  --model 'FieldForge (PT)' \
  --model Crystalite
```

它按 MP20 最常见空间群选择 novel、unique、stable/metastable 结构，复制 CIF，并生成结构图和 contact sheet。

## 配置

集中配置位于 `scripts/settings.py`。以下环境变量可覆盖默认路径：

| 环境变量 | 含义 |
|---|---|
| `CRYSTAL_BENCHMARK_DATA_DIR` | 数据根目录 |
| `CRYSTAL_BENCHMARK_OUTPUT_DIR` | 输出根目录 |
| `FIELD_FORGE_CIF_DIR` | FieldForge CIF 目录 |
| `CRYSTALITE_CIF_DIR` | Crystalite CIF 目录 |
| `MLIP_JSON_DIR` | 多 MLIP JSON 目录 |
| `MP20_CSV` | `mp20_total.csv` 路径 |

命令行参数优先用于单次实验；环境变量适合固定工作站或批处理作业。

## 可复现性约定

- 每个模型的 CIF 在词典序排序后取前 `--limit` 个；
- 缓存 CSV 保存逐结构结果，便于审计和无损重绘；
- 多 MLIP 图默认使用 `3 × IQR` 过滤异常值，原始值和被过滤值都保存在 CSV；
- stable：`mean_e_above_hull <= 0`；
- metastable：`0 < mean_e_above_hull <= 0.1 eV/atom`；
- 图像默认 600 dpi，并生成矢量 PDF 副本。

## 检查

无需数据即可检查语法和 CLI：

```bash
python -m compileall -q scripts tests
python scripts/run_all.py --help
```

安装开发依赖后可运行：

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

发布到 GitHub 前请自行选择并添加合适的开源许可证；本目录未擅自替项目指定许可证。
