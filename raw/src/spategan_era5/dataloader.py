"""
ERA5 数据集的数据加载工具。

提供加载数据集、规范化坐标名称以及将降水单位转换为 `mm/hr` 的函数。
"""

import numpy as np
import xarray as xr


# 降水变量名候选列表
CP_CANDIDATES = ['cp', 'convective', 'convcp', 'conv_precip', 'conv_precipitation']
LSP_CANDIDATES = ['lsp', 'ls_precip', 'large_scale', 'ls_precipitation', 'stratiform']


def detect_cp_lsp_vars(ds: xr.Dataset) -> list[str]:
    """
    检测 CP（对流降水）和 LSP（大尺度降水）的变量名。
    
    在数据集中搜索符合常见对流降水和大尺度降水命名模式的变量。
    
    参数：
        ds: 待搜索的 xarray Dataset。
        
    返回：
        按 [CP, LSP] 顺序返回找到的变量列表。
        
    抛出：
        ValueError: 找不到 CP 或 LSP 变量时抛出。
    """
    found_cp: str | None = None
    found_lsp: str | None = None

    for var in ds.data_vars:
        name = var.lower()
        if found_cp is None and any(c in name for c in CP_CANDIDATES):
            found_cp = var
        if found_lsp is None and any(c in name for c in LSP_CANDIDATES):
            found_lsp = var

    # 也接受完全匹配的大写名称
    if found_cp is None and 'CP' in ds.data_vars:
        found_cp = 'CP'
    if found_lsp is None and 'LSP' in ds.data_vars:
        found_lsp = 'LSP'

    # 检查两个变量是否都已找到
    available_vars = list(ds.data_vars)
    
    if found_cp is None and found_lsp is None:
        raise ValueError(
            f"Could not find CP (convective) or LSP (large-scale) precipitation variables. "
            f"Available variables: {available_vars}. "
            f"Expected CP names: {CP_CANDIDATES + ['CP']}. "
            f"Expected LSP names: {LSP_CANDIDATES + ['LSP']}."
        )
    elif found_cp is None:
        raise ValueError(
            f"Could not find CP (convective) precipitation variable. "
            f"Available variables: {available_vars}. "
            f"Expected names: {CP_CANDIDATES + ['CP']}."
        )
    elif found_lsp is None:
        raise ValueError(
            f"Could not find LSP (large-scale) precipitation variable. "
            f"Available variables: {available_vars}. "
            f"Expected names: {LSP_CANDIDATES + ['LSP']}."
        )
    
    return [found_cp, found_lsp]


# 单位转换系数
UNIT_CONVERSIONS: dict[str, float] = {
    'mm/hr': 1.0,
    'mm/h': 1.0,
    'mmperhr': 1.0,
    'mm/s': 3600.0,
    'mmsec': 3600.0,
    'm/s': 1000.0 * 3600.0,
    'msec': 3600.0 * 1000.0,
    'mpersec': 3600.0 * 1000.0,
    'm/hr': 1000.0,
    'm/h': 1000.0,
    'mperhr': 1000.0,
    'm': 1000.0,  # 假设为每小时米单位（ERA5 tp 累积量）
}


def _conversion_factor_to_mm_per_hr(unit: str) -> float:
    """
    计算将给定单位转换为 mm/hr 的乘法因子。
    
    参数：
        unit: 输入单位字符串（例如 'm/s'、'm/hr'、'mm/hr'）。
        
    返回：
        转换为 mm/hr 所需的乘法因子。
    """
    normalized_unit = unit.strip().lower().replace(' ', '')
    return UNIT_CONVERSIONS.get(normalized_unit, 1000.0)  # 默认兜底值


def normalize_longitude(
    ds: xr.Dataset
) -> xr.Dataset:
    """
    将经度坐标规范化到 -180..180 范围。
    
    参数
    ----------
    ds : xarray.Dataset
        包含经度坐标的输入数据集
   
    返回
    -------
    xarray.Dataset
        经度已规范化到 -180..180 的数据集
    """
    if 'lon' not in ds.coords:
        return ds
    
    lons = ds['lon'].values
    lon_min = np.nanmin(lons)
    lon_max = np.nanmax(lons)
    
    if lon_min < 0:
        # 包含负值，说明应为 -180..180
        is_0_360 = False
    elif lon_max > 180:
        # 数值超过 180，说明应为 0..360
        is_0_360 = True
    else:
        # 范围有歧义，默认按 -180..180 处理
        is_0_360 = False
    
    # 如有需要则转换
    if is_0_360:
        # 将 0..360 转换为 -180..180
        new_lons = np.where(lons > 180, lons - 360, lons)
        ds = ds.assign_coords(lon=new_lons)
        ds = ds.sortby('lon')
    else:
        # 已经是 -180..180，只需确保排序
        if not np.all(np.diff(lons) > 0):
            ds = ds.sortby('lon')
    
    return ds



def load_and_prepare_dataset(
    path: str,
    precip_unit: str = 'm',
    precip_var: str | None = None,
    required_hours: int = 16,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[xr.Dataset, str]:
    """
    从 `path` 加载 ERA5 数据集，规范化坐标名称，可选按日期范围筛选，
    选择前 `required_hours` 小时（或日期范围内的数据），检测降水变量
    （除非显式提供），并将单位转换为 `mm/hr`。

    Parameters:
        path: netCDF 数据集文件路径
        precip_unit: 输入数据中的降水单位
        precip_var: 指定降水变量名（为 None 时自动检测）
        required_hours: 需要提取的小时数
        start_date: 筛选起始日期（格式：'YYYY-MM-DD'，可选）
        end_date: 筛选结束日期（格式：'YYYY-MM-DD'，可选）

    返回：
        ds_prepared, precip_var_name
    """
    ds = xr.open_dataset(path)

    # 重命名维度名称
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon'})

    # 规范化经度
    ds = normalize_longitude(ds)

    if 'lat' not in ds.coords or 'lon' not in ds.coords:
        raise ValueError(
            "输入数据必须包含经纬度坐标 lat/lon（或 latitude/longitude）。"
            "main.py 需要原始 ERA5 经纬度 NetCDF 文件；data/x_test.nc 是 notebook "
            "示例中已经投影到 y/x 网格的模型输入，不能直接作为 main.py 的输入。"
        )

    # 检测 CP/LSP 变量（优先使用 CP 和 LSP 通道）
    precip_vars = detect_cp_lsp_vars(ds)

    # 确保存在 time 维度
    if 'time' not in ds.dims:
        raise ValueError("Dataset must contain a 'time' dimension")
    ds_original_start = ds.time.values[0]
    ds_original_end = ds.time.values[-1]

    # 如果提供日期范围，则按日期范围筛选
    if start_date is not None or end_date is not None:
        ds = ds.sel(time=slice(start_date, end_date))
        if len(ds.time) == 0:
            available_start = ds_original_start
            available_end = ds_original_end
            raise ValueError(
                "指定时间范围内没有数据："
                f"{start_date} 到 {end_date}。"
                f"输入文件可用时间范围是 {available_start} 到 {available_end}。"
            )
    
    # 如果时间长度足够，则切片到所需小时数
    if len(ds.time) < required_hours:
        raise ValueError(f"Dataset has {len(ds.time)} time steps but {required_hours} are required")

    # 仅保留检测到的降水变量（丢弃其它变量）
    # 按 CP、LSP 的顺序排列
    ds = ds[[v for v in precip_vars if v in ds.data_vars]]

    # 将每个降水变量的单位转换为 mm/hr
    factor = _conversion_factor_to_mm_per_hr(precip_unit)
    for v in list(ds.data_vars):
        ds[v] = ds[v] * factor
        ds[v].attrs['units'] = 'mm/hr'

    # 附加检测到的变量列表，供需要两个通道的调用方使用
    ds.attrs['precip_variables'] = precip_vars

    # 返回主变量名（第一个检测到的变量）
    primary = precip_vars[0] if precip_vars else list(ds.data_vars)[0]
    return ds, primary
