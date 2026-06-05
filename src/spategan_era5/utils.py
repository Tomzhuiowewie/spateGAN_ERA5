"""
spateGAN-ERA5 的工具函数。

包含距离计算、插值和文件名生成等辅助函数。
"""

import math
from pathlib import Path

import pandas as pd
import torch.nn as nn

# 地球半径，单位为千米（WGS84 平均半径）
EARTH_RADIUS_KM = 6371.04


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算地球上两点之间的大圆距离。
    
    使用 Haversine 公式在球面上进行较准确的距离计算。
    
    参数：
        lat1: 第一个点的纬度，单位为度。
        lon1: 第一个点的经度，单位为度。
        lat2: 第二个点的纬度，单位为度。
        lon2: 第二个点的经度，单位为度。
        
    返回：
        两点之间的距离，单位为千米。
    """

    # 将纬度和经度从角度转换为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 计算差值
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine 公式
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 距离，单位为千米
    distance = EARTH_RADIUS_KM * c

    return distance



class DataInterpolation(nn.Module):
    """用于插值 5D 张量数据的 PyTorch 模块。
    
    支持对 batch、channel、time、height、width 张量进行
    2D（bicubic/bilinear/nearest）和 3D（trilinear）插值。
    
    参数：
        size: 目标输出尺寸（height, width）或（time, height, width）。
        mode: 插值模式（'bicubic'、'bilinear'、'nearest'、'trilinear'）。
        corners: 插值时是否对齐角点。
        antialias: 是否应用抗锯齿。
    """
    
    def __init__(
        self,
        size: tuple[int, int] | tuple[int, int, int],
        mode: str,
        corners: bool | None = None,
        antialias: bool | None = None,
    ) -> None:
        super().__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.align_corners = corners
        self.antialias = antialias

    def forward(self, x):
        """对输入张量进行插值。
        
        参数：
            x: 形状为 (B, C, T, H, W) 的输入张量。
            
        返回：
            插值后的张量。
        """
        if self.mode in ('bicubic', 'bilinear', 'nearest'):
            b, c, t, w, h = x.size()
            x = x.reshape(b, c * t, w, h)
            x = self.interp(
                x,
                size=self.size,
                mode=self.mode,
                align_corners=self.align_corners,
                antialias=self.antialias,
            )
            x = x.reshape(b, c, t, int(self.size[0]), int(self.size[1]))
        else:
            x = self.interp(
                x,
                size=self.size,
                mode=self.mode,
                align_corners=self.align_corners,
                antialias=self.antialias,
            )
        return x
    

def generate_output_filename(
    dataset: "xr.Dataset",
    config: dict,
    projection: str = 'latlon',
    model: str = 'spateGAN_ERA5',
) -> str:
    """
    根据参数生成输出文件名。
    
    参数
    ----------
    dataset: xr.Dataset
        包含时间和坐标信息的数据集
    projection : str
        投影类型（'latlon' 或 'utm'）
    
    返回
    -------
    str
        格式化后的文件名
    """
    
    center_lat = dataset.attrs['center_lat']
    center_lon = dataset.attrs['center_lon'] 
    start_date = dataset.time.values[0]
    end_date = dataset.time.values[-1]

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    # 格式化带方向标记的纬度/经度
    lat_str = f"{abs(center_lat):.2f}{'N' if center_lat >= 0 else 'S'}"
    lon_str = f"{abs(center_lon):.2f}{'E' if center_lon >= 0 else 'W'}"
    
    # 格式化日期
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    if model == 'spateGAN_ERA5':
        filename = f"{model}_{projection}_{lat_str}_{lon_str}_{start_str}_{end_str}_e{config['processing']['seed']}.nc"
    elif model == 'era5':
        filename = f"{model}_{projection}_{lat_str}_{lon_str}_{start_str}_{end_str}.nc"
    else:
        raise ValueError(f"Unknown model type: {model}")
    
    return filename
