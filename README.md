## spateGAN_ERA5

**spateGAN-ERA5** 是一个用于 ERA5 降水数据时空降尺度的深度学习框架。它使用概率条件生成对抗网络（cGANs），将 ERA5 数据从 24 km、1 小时间隔提升到 2 km、10 分钟间隔。该方法能够生成具有真实时空模式和准确雨强分布的高分辨率降雨场，并能覆盖极端降水事件。（https://www.nature.com/articles/s41612-025-01103-y）

---

### 功能特点

- **全球泛化能力**使用德国雨量计校正雷达数据训练，并在美国、澳大利亚等区域进行了验证。
- **高分辨率输出**将粗分辨率 ERA5 降水数据转换为 2 km × 10 min 的降水场，可用于洪水与水文风险评估。
- **不确定性量化**可生成多样化的降水情景集合。
- **用户自定义推理**
  通过简单的配置驱动流程，使用自有 ERA5 netCDF 文件，在自定义区域和时间段上运行 spateGAN-ERA5。

---

### 数据集

训练、模型选择和评估数据集可从 https://doi.org/10.5281/zenodo.17417589 获取。

如需在推理阶段对降水数据进行降尺度，可从[这里](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download)下载小时分辨率 ERA5 数据，并保存为 netCDF 文件。
需要的变量包括 "Convective precipitation" 和 "Large scale precipitation"。此外，空间范围至少需要 672 km × 672 km，时间序列长度至少需要 16 小时。

---

### 快速开始

#### 安装

**conda**

```bash
mamba env create -f environment.yaml
or
conda env create -f environment.yaml
```

**uv**
克隆仓库。
安装 uv：https://docs.astral.sh/uv/getting-started/installation/

```bash
cd spateGAN_ERA5
uv sync # 创建 .venv
```

使用 `spateGAN_ERA5/.venv/bin/python3` 作为 Python 解释器。

#### 降尺度

更新 `config.yaml`，设置**降尺度区域（用中心点经纬度坐标定义）**、ERA5 netCDF 输入路径和输出路径。

运行：

```bash
uv run main.py
```

高分辨率降水场会分别以 UTM 投影（2 km、10 min 分辨率）和经纬度网格（0.018°、10 min 分辨率）保存。

#### 概率降尺度

如需进行概率降尺度，可修改 `config.yaml` 中的 `seed` 和 `slide` 参数。

#### 评估示例

仓库中包含一个用于评估示例的 notebook：

downscaling.ipynb（以及 downscaling_example.py）使用一个小型样例数据集，演示如何用 spateGAN-ERA5 对德国区域的 ERA5 降水进行降尺度，并与雷达观测进行对比。
