"""spateGAN-ERA5：用于 ERA5 降水降尺度的深度学习框架。

本包提供基于概率条件 GAN 架构的工具，用于将 ERA5 降水数据从
24 km/1 小时分辨率时空降尺度到 2 km/10 分钟分辨率。

用法示例：
    >>> from spategan_era5 import Generator, ERA5DownscalingInference
    >>> from spategan_era5.dataloader import load_and_prepare_dataset
    >>> from spategan_era5.projection import latlon_to_utm, utm_to_latlon
    >>> from spategan_era5.pipeline import run_downscaling_pipeline
"""

from .dataloader import (
    detect_cp_lsp_vars,
    load_and_prepare_dataset,
    normalize_longitude,
)
from .downscaling_inference import ERA5DownscalingInference
from .inference import InferenceEngine
from .model import Generator
from .pipeline import run_downscaling_pipeline
from .preprocessing import (
    calculate_domain_center,
    slice_data_for_projection,
    validate_patch_extraction,
    validate_time_dimension,
)
from .projection import (
    latlon_to_utm,
    prediction_output_dataset,
    utm_to_latlon,
)
from .utils import (
    DataInterpolation,
    generate_output_filename,
    haversine,
)

__all__ = [
    # 模型
    "Generator",
    # 推理
    "InferenceEngine",
    "ERA5DownscalingInference",
    # 流水线
    "run_downscaling_pipeline",
    # 数据加载
    "load_and_prepare_dataset",
    "detect_cp_lsp_vars",
    "normalize_longitude",
    # 预处理
    "validate_patch_extraction",
    "slice_data_for_projection",
    "calculate_domain_center",
    "validate_time_dimension",
    # 投影
    "latlon_to_utm",
    "utm_to_latlon",
    "prediction_output_dataset",
    # 工具函数
    "haversine",
    "DataInterpolation",
    "generate_output_filename",
]

__version__ = "0.1.0"
