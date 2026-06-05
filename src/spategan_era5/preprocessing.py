"""
ERA5 数据校验和切片的预处理模块。

处理区域大小校验和数据提取。
"""

import numpy as np
import xarray as xr
from typing import Tuple

from utils import haversine


def validate_patch_extraction(
    center_lat: float,
    center_lon: float,
    patch_size_km: float,
    patch_padding_km: float,
    lat_south: float,
    lat_north: float,
    lon_west: float,
    lon_east: float,
) -> dict[str, float]:
    """
    校验数据区域是否足够大，能够提取请求的 patch。
    
    patch 以 (center_lat, center_lon) 为中心提取，在各方向上的半径为
    (patch_size_km/2 + patch_padding_km)，用于补偿经纬度畸变并确保 UTM
    投影后的覆盖范围。
    
    参数：
        center_lat: 提取中心纬度（度）
        center_lon: 提取中心经度（度）
        patch_size_km: 基础 patch 大小，单位为 km（例如 672）
        patch_padding_km: 用于畸变补偿的额外填充（例如 100）
        lat_south: 可用数据南边界（度）
        lat_north: 可用数据北边界（度）
        lon_west: 可用数据西边界（度）
        lon_east: 可用数据东边界（度）
    
    返回：
        包含校验结果的字典
    
    抛出：
        ValueError: 无法从可用数据中提取 patch 时抛出
    """
    # 计算从中心开始所需的半径
    half_patch_km = patch_size_km / 2.0
    required_radius_km = half_patch_km + patch_padding_km
    
    errors = []
    
    # 先检查中心点是否位于数据区域内
    if not (lat_south <= center_lat <= lat_north):
        errors.append(f"Center latitude {center_lat:.2f}° is outside data range [{lat_south:.2f}°, {lat_north:.2f}°]")
    
    # 对经度处理全球数据的环绕情况
    # 规范化到同一约定以便比较
    center_lon_norm = ((center_lon + 180) % 360) - 180
    lon_west_norm = ((lon_west + 180) % 360) - 180
    lon_east_norm = ((lon_east + 180) % 360) - 180
    
    # 检查是否为全球经度数据
    is_global_lon = abs(lon_east - lon_west) > 350
    
    if not is_global_lon:
        # 区域数据：中心点必须位于边界内
        if lon_west_norm <= lon_east_norm:
            # 普通情况
            if not (lon_west_norm <= center_lon_norm <= lon_east_norm):
                errors.append(f"Center longitude {center_lon:.2f}° is outside data range [{lon_west:.2f}°, {lon_east:.2f}°]")
        else:
            # 跨越 180° 经线
            if not (center_lon_norm >= lon_west_norm or center_lon_norm <= lon_east_norm):
                errors.append(f"Center longitude {center_lon:.2f}° is outside data range [{lon_west:.2f}°, {lon_east:.2f}°]")
    
    if center_lon > 180:
        errors.append(f"Center longitude {center_lon:.2f}° exceeds 180°. Please use -180 to 180 range.")
    
    # 如果中心点在数据范围外，立即失败
    if errors:
        error_msg = "Cannot extract patch:\n  " + "\n  ".join(errors)
        raise ValueError(error_msg)
    
    # 检查中心点到数据边界的距离覆盖
    lat_distance_south = haversine(center_lat, 0, lat_south, 0)
    lat_distance_north = haversine(center_lat, 0, lat_north, 0)
    
    # 检查经度覆盖（在中心纬度处）
    lon_distance_west = haversine(center_lat, center_lon, center_lat, lon_west)
    lon_distance_east = haversine(center_lat, center_lon, center_lat, lon_east)
    
    # 校验覆盖范围
    if lat_distance_south < required_radius_km:
        errors.append(f"Insufficient coverage south: {lat_distance_south:.2f} km < {required_radius_km:.2f} km")
    if lat_distance_north < required_radius_km:
        errors.append(f"Insufficient coverage north: {lat_distance_north:.2f} km < {required_radius_km:.2f} km")
    if lon_distance_west < required_radius_km:
        errors.append(f"Insufficient coverage west: {lon_distance_west:.2f} km < {required_radius_km:.2f} km")
    if lon_distance_east < required_radius_km:
        errors.append(f"Insufficient coverage east: {lon_distance_east:.2f} km < {required_radius_km:.2f} km")
    
    if errors:
        error_msg = "Cannot extract patch:\n  " + "\n  ".join(errors)
        raise ValueError(error_msg)
    
    print(f"✓ patch validated: center ({center_lat:.2f}°N, {center_lon:.2f}°E), radius {required_radius_km:.0f} km")
    
    return {
        "lat_distance_south": lat_distance_south,
        "lat_distance_north": lat_distance_north,
        "lon_distance_west": lon_distance_west,
        "lon_distance_east": lon_distance_east,
        "required_radius_km": required_radius_km
    }


def slice_data_for_projection(
    ds: xr.Dataset,
    center_lat: float,
    center_lon: float,
    target_domain_size_km: float,
    extra_padding_cells: int = 2,
    era5_grid_size_km: float = 25.0,
) -> tuple[xr.Dataset, dict]:
    """
    在经纬度投影中切片 ERA5 数据，提取居中的 patch。
    
    以指定坐标为中心提取一个略大于目标区域大小（672×672 km）的 patch。
    该步骤在任何 UTM 转换之前，于经纬度投影中完成。
    
    使用 Haversine 距离计算所需像素数，并考虑经度间距随纬度变化的特性。
    支持处理具有经度环绕的全球数据（0-360 或 -180-180）。
    
    策略：
    1. 在中心坐标处使用 Haversine 计算实际经纬度间距
    2. 纬度方向：使用 center_lat 处的间距
    3. 经度方向：使用 center_lat 处的间距（经度压缩与纬度相关）
    4. 添加填充：pixels = ceil(domain_km/2 / spacing) * 2 + padding
    5. 提取以 (center_lat, center_lon) 为中心的矩形 patch
    
    参数：
        ds: 带经纬度坐标的输入 xarray Dataset
        center_lat: 提取中心纬度（度）
        center_lon: 提取中心经度（度）
        target_domain_size_km: 目标区域大小，单位 km（例如 672）
        extra_padding_cells: 额外填充单元数（默认 2）
        era5_grid_size_km: ERA5 基准网格分辨率，单位 km（约 25 km），作为兜底值
    
    返回：
        (sliced_dataset, slicing_info_dict) 元组
        - sliced_dataset: 形状为 (N_lat, N_lon, time) 的 xr.Dataset
        - slicing_info_dict: 包含提取元数据的字典
    """
    # 获取坐标数组
    lat_coords = ds.coords['lat'].values
    lon_coords = ds.coords['lon'].values
    
    # 查找最接近中心坐标的索引
    center_lat_idx = np.argmin(np.abs(lat_coords - center_lat))
    center_lon_idx = np.argmin(np.abs(lon_coords - center_lon))
    
    # 半区域大小，单位 km
    half_domain_km = target_domain_size_km / 2.0
    
    # ===== 纬度间距 =====
    # 计算中心附近相邻纬度索引之间的实际间距
    if center_lat_idx > 0 and center_lat_idx < len(lat_coords) - 1:
        lat1 = lat_coords[center_lat_idx]
        lat2 = lat_coords[center_lat_idx - 1]
        lat_spacing_km = haversine(lat1, 0, lat2, 0)
    else:
        # 兜底：使用可用相邻索引处的间距
        if center_lat_idx < len(lat_coords) - 1:
            lat1 = lat_coords[center_lat_idx]
            lat2 = lat_coords[center_lat_idx + 1]
            lat_spacing_km = haversine(lat1, 0, lat2, 0)
        else:
            lat_spacing_km = era5_grid_size_km
    
    # 所需纬度像素数（从中心向每个方向）
    lat_pixels_half = int(np.ceil(half_domain_km / lat_spacing_km))
    # 填充应用于两侧（上、下）
    lat_pixels_total = lat_pixels_half * 2 + extra_padding_cells * 2
    
    # ===== 经度间距 =====
    # 计算 center_lat 处相邻经度索引之间的实际间距
    # （经度间距依赖纬度，即 cos(lat) 因子）
    if center_lon_idx > 0 and center_lon_idx < len(lon_coords) - 1:
        lon1 = lon_coords[center_lon_idx]
        lon2 = lon_coords[center_lon_idx - 1]
        lon_spacing_km = haversine(center_lat, lon1, center_lat, lon2)
    else:
        # 兜底：使用可用相邻索引处的间距
        if center_lon_idx < len(lon_coords) - 1:
            lon1 = lon_coords[center_lon_idx]
            lon2 = lon_coords[center_lon_idx + 1]
            lon_spacing_km = haversine(center_lat, lon1, center_lat, lon2)
        else:
            lon_spacing_km = era5_grid_size_km * np.cos(np.radians(center_lat))
    
    # 避免在极区出现除零
    if lon_spacing_km < 1e-6:
        lon_spacing_km = 0.1  # 接近极点时使用一个小的正值
    
    # 所需经度像素数（从中心向每个方向）
    lon_pixels_half = int(np.ceil(half_domain_km / lon_spacing_km))
    # 填充应用于两侧（左、右）
    lon_pixels_total = lon_pixels_half * 2 + extra_padding_cells * 2
    
    # ===== 提取 PATCH =====
    # 计算纬度方向的切片索引（直接处理）
    # 在两侧应用填充
    lat_idx_start = center_lat_idx - lat_pixels_half - extra_padding_cells
    lat_idx_end = center_lat_idx + lat_pixels_half + extra_padding_cells + (lat_pixels_total % 2)
    
    # 将纬度限制到有效范围
    lat_idx_start = max(0, lat_idx_start)
    lat_idx_end = min(len(lat_coords), lat_idx_end)
    
    # 对经度处理全球环绕（适用于 lon 0-360 或 -180-180 的数据）
    is_global_lon = (len(lon_coords) > 100) and (lon_coords[-1] - lon_coords[0] > 350)
    
    if is_global_lon:
        # 全球数据：允许环绕，并在两侧应用填充
        lon_idx_start = (center_lon_idx - lon_pixels_half - extra_padding_cells) % len(lon_coords)
        lon_idx_end = (center_lon_idx + lon_pixels_half + extra_padding_cells + (lon_pixels_total % 2)) % len(lon_coords)
        
        if lon_idx_end < lon_idx_start:
            # 跨越日期变更线
            ds_sliced = xr.concat([
                ds.isel(lat=slice(lat_idx_start, lat_idx_end),
                       lon=slice(lon_idx_start, None)),
                ds.isel(lat=slice(lat_idx_start, lat_idx_end),
                       lon=slice(None, lon_idx_end))
            ], dim='lon')
        else:
            ds_sliced = ds.isel(lat=slice(lat_idx_start, lat_idx_end),
                               lon=slice(lon_idx_start, lon_idx_end))
    else:
        # 有限经度范围：不环绕，直接裁剪到边界
        lon_idx_start = max(0, center_lon_idx - lon_pixels_half - extra_padding_cells)
        lon_idx_end = min(len(lon_coords), center_lon_idx + lon_pixels_half + extra_padding_cells + (lon_pixels_total % 2))
        
        ds_sliced = ds.isel(lat=slice(lat_idx_start, lat_idx_end),
                           lon=slice(lon_idx_start, lon_idx_end))
    
    # 汇总切片信息
    slicing_info = {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "center_lat_idx": center_lat_idx,
        "center_lon_idx": center_lon_idx,
        "lat_indices": (lat_idx_start, lat_idx_end),
        "lon_indices": (lon_idx_start, lon_idx_end) if not is_global_lon or lon_idx_end >= lon_idx_start else "wrapped",
        "extracted_lat_range": (float(ds_sliced.lat.min()), float(ds_sliced.lat.max())),
        "extracted_lon_range": (float(ds_sliced.lon.min()), float(ds_sliced.lon.max())),
        "extracted_shape": (len(ds_sliced.lat), len(ds_sliced.lon)),
        "lat_pixels_total": lat_pixels_total,
        "lon_pixels_total": lon_pixels_total,
        "lat_spacing_km": lat_spacing_km,
        "lon_spacing_km": lon_spacing_km,
        "target_domain_size_km": target_domain_size_km,
        "is_global_data": is_global_lon,
    }
    
    return ds_sliced, slicing_info


def calculate_domain_center(ds: xr.Dataset) -> Tuple[float, float]:
    """
    计算切片区域的中心坐标。
    
    使用给定数据集坐标的地理中心。
    
    参数：
        ds: 带经纬度坐标的 xarray Dataset（通常是已切片数据）
    
    返回：
        (center_lat, center_lon) 元组
    """
    lat_coords = ds.coords['lat'].values
    lon_coords = ds.coords['lon'].values
    
    center_lat = (lat_coords.min() + lat_coords.max()) / 2.0
    center_lon = (lon_coords.min() + lon_coords.max()) / 2.0
    
    return center_lat, center_lon


def validate_time_dimension(ds: xr.Dataset, required_hours: int = 16) -> bool:
    """
    校验数据集是否包含足够的时间步。
    
    对于 16 小时需求，需要 16 个逐小时时间步。
    
    参数：
        ds: 输入数据集
        required_hours: 所需小时数
    
    返回：
        有效时返回 True，否则抛出 ValueError
    
    抛出：
        ValueError: 时间步不足时抛出
    """
    if 'time' not in ds.dims:
        raise ValueError("Dataset must have a 'time' dimension")
    
    n_time_steps = len(ds.time)
    
    if n_time_steps < required_hours:
        raise ValueError(
            f"Dataset has {n_time_steps} time steps but {required_hours} are required"
        )
    
    # 针对大数据集发出警告（>1 个月，约 720 小时）
    if n_time_steps > 720:
        print(f"⚠ Warning: Large dataset with {n_time_steps} time steps (~{n_time_steps/24:.1f} days) may cause memory issues")
    
    return True


def extract_time_window(
    ds: xr.Dataset,
    start_idx: int,
    n_hours: int
) -> xr.Dataset:
    """
    从数据集中提取指定时间窗口。
    
    参数：
        ds: 输入数据集
        start_idx: 起始时间索引
        n_hours: 要提取的小时数
    
    返回：
        包含指定时间窗口的切片数据集
    """
    return ds.isel(time=slice(start_idx, start_idx + n_hours))
