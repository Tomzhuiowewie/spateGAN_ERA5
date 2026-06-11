"""
ERA5 数据的投影与重投影模块。

处理经纬度投影与 UTM 投影之间的转换。
"""

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator




def get_utm_crs(utm_zone: int, center_lat: float) -> CRS:
    """
    获取 UTM 投影对应的 CRS（坐标参考系统）。
    
    参数：
        utm_zone: UTM 分区编号
        center_lat: 中心纬度（用于判断南北半球）
    
    返回：
        pyproj CRS 对象
    """
    # 判断半球
    if center_lat >= 0:
        epsg_code = 32600 + utm_zone  # 北半球
    else:
        epsg_code = 32700 + utm_zone  # 南半球
    
    crs = CRS.from_epsg(epsg_code)
    return crs


def latlon_to_utm(ds_sliced, center_lat, center_lon, target_size, grid_spacing, method='bilinear'):
    """
    将 xarray 数据集从经纬度投影转换到 UTM 投影。
    
    参数
    ----------
    ds_sliced : xarray.Dataset or xarray.DataArray
        维度为 (time, lat, lon) 或 (lat, lon) 的输入数据集
    center_lat : float
        中心纬度（例如 22.6250）
    center_lon : float
        中心经度（例如 96.3750）
    target_size : int
        目标网格像素大小（例如 336 表示 336x336，28 表示 28x28）
    grid_spacing : float
        网格间距，单位为米（例如 2000 表示 2 km，24000 表示 24 km）
    method : str
        插值方法：'bilinear'（或 'linear'）或 'nearest'
    
    返回
    -------
    xarray.Dataset
        带 UTM 坐标的转换后数据集
    """
    
    # 映射方法名称
    if method == 'bilinear':
        method = 'linear'
    elif method not in ['linear', 'nearest']:
        raise ValueError(f"Unknown interpolation method: {method}. Use 'bilinear' or 'nearest'.")
    
    # 将中心经度规范化到 [-180, 180)，确保无论输入经度采用何种约定，
    # UTM 分区计算和重投影都保持一致。
    center_lon = ((center_lon + 180) % 360) - 180
    utm_zone = int((center_lon + 180) / 6) + 1
    hemisphere = 'north' if center_lat >= 0 else 'south'
    
    # 创建 CRS 对象
    crs_latlon = CRS.from_epsg(4326)  # WGS84
    crs_utm = CRS.from_proj4(f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84")
    
    # 创建坐标转换器
    transformer_to_utm = Transformer.from_crs(crs_latlon, crs_utm, always_xy=True)
    transformer_to_latlon = Transformer.from_crs(crs_utm, crs_latlon, always_xy=True)
    
    # 处理 DataArray 输入
    if isinstance(ds_sliced, xr.DataArray):
        var_name = ds_sliced.name or 'data'
        ds_sliced = ds_sliced.to_dataset(name=var_name)
    
    # 获取原始坐标
    lats = ds_sliced.lat.values
    lons = ds_sliced.lon.values
    
    # 将中心点转换到 UTM（center_lon 已规范化）
    center_x, center_y = transformer_to_utm.transform(center_lon, center_lat)
    
    # 定义目标 UTM 网格（以中心点为中心）
    half_extent = (target_size * grid_spacing) / 2
    
    x_target = np.arange(
        center_x - half_extent + grid_spacing / 2,
        center_x + half_extent,
        grid_spacing
    )
    y_target = np.arange(
        center_y - half_extent + grid_spacing / 2,
        center_y + half_extent,
        grid_spacing
    )
    
    # 确保像素数精确等于 target_size
    x_target = x_target[:target_size]
    y_target = y_target[:target_size]
    
    # 创建目标网格
    x_target_grid, y_target_grid = np.meshgrid(x_target, y_target)
    
    # 将目标 UTM 网格转回经纬度，用于插值
    lon_target, lat_target = transformer_to_latlon.transform(
        x_target_grid.ravel(), 
        y_target_grid.ravel()
    )
    lon_target = lon_target.reshape(x_target_grid.shape)
    lat_target = lat_target.reshape(y_target_grid.shape)
    
    # 获取数据变量名
    data_vars = list(ds_sliced.data_vars)
    
    # 检查是否存在 time 维度
    has_time = 'time' in ds_sliced.dims
    
    # 为向量化操作准备插值器
    data_dict = {}
    points = np.column_stack([lat_target.ravel(), lon_target.ravel()])
    
    for var in data_vars:
        if has_time:
            # 沿时间维度进行向量化插值
            data_3d = ds_sliced[var].values  # 形状：(time, lat, lon)
            
            # 为整个 3D 数据创建插值器
            interpolator = RegularGridInterpolator(
                (np.arange(data_3d.shape[0]), lats, lons),
                data_3d,
                method=method,
                bounds_error=False,
                fill_value=np.nan
            )
            
            # 创建带时间维度的点数组（为所有时间复制）
            n_times = len(ds_sliced.time)
            t_indices = np.repeat(np.arange(n_times), len(points))
            points_with_time = np.column_stack([
                t_indices,
                np.tile(points[:, 0], n_times),
                np.tile(points[:, 1], n_times)
            ])
            
            # 一次性插值所有时间
            interpolated = interpolator(points_with_time).reshape(n_times, target_size, target_size)
            data_dict[var] = (['time', 'y', 'x'], interpolated)
        else:
            # 单个时间步（无 time 维度）
            data = ds_sliced[var].values
            interpolator = RegularGridInterpolator(
                (lats, lons),
                data,
                method=method,
                bounds_error=False,
                fill_value=np.nan
            )
            interpolated = interpolator(points).reshape(target_size, target_size)
            data_dict[var] = (['y', 'x'], interpolated)
    
    coords = {
        'x': x_target,
        'y': y_target,
    }
    if has_time:
        coords['time'] = ds_sliced.time.values
    
    ds_utm = xr.Dataset(data_dict, coords=coords)
    
    # 添加属性
    ds_utm.attrs['crs'] = crs_utm.to_string()
    ds_utm.attrs['utm_zone'] = utm_zone
    ds_utm.attrs['center_lat'] = center_lat
    ds_utm.attrs['center_lon'] = center_lon
    ds_utm.attrs['grid_spacing_m'] = grid_spacing
    ds_utm.attrs['target_size'] = target_size
    ds_utm.x.attrs['units'] = 'meters'
    ds_utm.x.attrs['long_name'] = 'UTM easting'
    ds_utm.y.attrs['units'] = 'meters'
    ds_utm.y.attrs['long_name'] = 'UTM northing'
    
    # 为降水变量添加单位
    for var in data_vars:
        if 'cp' in var.lower() or 'lsp' in var.lower() or 'precip' in var.lower():
            ds_utm[var].attrs['units'] = 'mm/h'
            ds_utm[var].attrs['long_name'] = 'precipitation'
    
    return ds_utm




def utm_to_latlon(ds_utm, target_lats=None, target_lons=None, resolution=None, method='nearest'):
    """
    将 xarray 数据集从 UTM 投影转换回经纬度投影。
    
    参数
    ----------
    ds_utm : xarray.Dataset
        带 UTM 坐标 (x, y) 和必要属性的输入数据集
        (crs, center_lat, center_lon, utm_zone)
    target_lats : array-like, optional
        目标纬度坐标。为 None 时根据 UTM 范围计算。
    target_lons : array-like, optional
        目标经度坐标。为 None 时根据 UTM 范围计算。
    resolution : float, optional
        目标分辨率，单位为度（例如 0.25 表示 0.25° 网格间距）。
        在未提供 target_lats/target_lons 时使用。
    method : str
        插值方法（'linear'、'nearest'、'cubic'）
    
    返回
    -------
    xarray.Dataset
        带经纬度坐标的转换后数据集
    """
    
    # 从 UTM 数据集中提取属性
    utm_zone = ds_utm.attrs.get('utm_zone')
    center_lat = ds_utm.attrs.get('center_lat')
    center_lon = ds_utm.attrs.get('center_lon')
    crs_string = ds_utm.attrs.get('crs')
    
    if utm_zone is None or center_lat is None:
        raise ValueError("Dataset missing required attributes (utm_zone, center_lat, center_lon)")
    
    # 判断半球
    hemisphere = 'north' if center_lat >= 0 else 'south'
    
    # 创建 CRS 对象
    crs_latlon = CRS.from_epsg(4326)  # WGS84
    if crs_string:
        crs_utm = CRS.from_string(crs_string)
    else:
        crs_utm = CRS.from_proj4(f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84")
    
    # 创建坐标转换器
    transformer_to_latlon = Transformer.from_crs(crs_utm, crs_latlon, always_xy=True)
    transformer_to_utm = Transformer.from_crs(crs_latlon, crs_utm, always_xy=True)
    
    # 获取 UTM 坐标
    x_utm = ds_utm.x.values
    y_utm = ds_utm.y.values
    
    # 转换四个角点和边界，以获取准确的经纬度范围
    n_edge_points = 50
    
    # 沿所有边界采样
    edge_x_bottom = np.linspace(x_utm[0], x_utm[-1], n_edge_points)
    edge_y_bottom = np.full(n_edge_points, y_utm[0])
    edge_x_top = np.linspace(x_utm[0], x_utm[-1], n_edge_points)
    edge_y_top = np.full(n_edge_points, y_utm[-1])
    edge_x_left = np.full(n_edge_points, x_utm[0])
    edge_y_left = np.linspace(y_utm[0], y_utm[-1], n_edge_points)
    edge_x_right = np.full(n_edge_points, x_utm[-1])
    edge_y_right = np.linspace(y_utm[0], y_utm[-1], n_edge_points)
    
    all_edge_x = np.concatenate([edge_x_bottom, edge_x_top, edge_x_left, edge_x_right])
    all_edge_y = np.concatenate([edge_y_bottom, edge_y_top, edge_y_left, edge_y_right])
    
    all_edge_lon, all_edge_lat = transformer_to_latlon.transform(all_edge_x, all_edge_y)
    
    lat_min, lat_max = all_edge_lat.min(), all_edge_lat.max()
    lon_min, lon_max = all_edge_lon.min(), all_edge_lon.max()
    
    # 如果未提供目标经纬度网格，则计算它
    if target_lats is None or target_lons is None:
        if resolution is None:
            raise ValueError("Either provide target_lats/target_lons or specify resolution in degrees")
        
        if target_lats is None:
            # 对齐到指定分辨率的网格
            lat_start = np.floor(lat_min / resolution) * resolution
            lat_end = np.ceil(lat_max / resolution) * resolution
            target_lats = np.arange(lat_start, lat_end + resolution / 2, resolution)
            # 裁剪到实际范围
            target_lats = target_lats[(target_lats >= lat_min) & (target_lats <= lat_max)]
        
        if target_lons is None:
            # 对齐到指定分辨率的网格
            lon_start = np.floor(lon_min / resolution) * resolution
            lon_end = np.ceil(lon_max / resolution) * resolution
            target_lons = np.arange(lon_start, lon_end + resolution / 2, resolution)
            # 裁剪到实际范围
            target_lons = target_lons[(target_lons >= lon_min) & (target_lons <= lon_max)]
    
    # 创建目标经纬度网格
    lon_target, lat_target = np.meshgrid(target_lons, target_lats)
    
    # 将目标经纬度转换到 UTM，用于插值
    x_target, y_target = transformer_to_utm.transform(lon_target.ravel(), lat_target.ravel())
    x_target = x_target.reshape(lon_target.shape)
    y_target = y_target.reshape(lat_target.shape)
    
    # 获取数据变量名
    data_vars = list(ds_utm.data_vars)
    
    # 检查是否存在 time 维度
    has_time = 'time' in ds_utm.dims
    
    # 为向量化操作准备插值器
    data_dict = {}
    points = np.column_stack([y_target.ravel(), x_target.ravel()])
    
    for var in data_vars:
        if has_time:
            # 沿时间维度进行向量化插值
            data_3d = ds_utm[var].values  # 形状：(time, y, x)
            
            # 为整个 3D 数据创建插值器
            interpolator = RegularGridInterpolator(
                (np.arange(data_3d.shape[0]), y_utm, x_utm),
                data_3d,
                method=method,
                bounds_error=False,
                fill_value=np.nan
            )
            
            # 创建带时间维度的点数组（为所有时间复制）
            n_times = data_3d.shape[0]
            t_indices = np.repeat(np.arange(n_times), len(points))
            points_with_time = np.column_stack([
                t_indices,
                np.tile(points[:, 0], n_times),
                np.tile(points[:, 1], n_times)
            ])
            
            # 一次性插值所有时间
            interpolated = interpolator(points_with_time).reshape(n_times, len(target_lats), len(target_lons))
            data_dict[var] = (['time', 'lat', 'lon'], interpolated)
        else:
            # 单个时间步（无 time 维度）
            data = ds_utm[var].values
            interpolator = RegularGridInterpolator(
                (y_utm, x_utm),
                data,
                method=method,
                bounds_error=False,
                fill_value=np.nan
            )
            interpolated = interpolator(points).reshape(len(target_lats), len(target_lons))
            data_dict[var] = (['lat', 'lon'], interpolated)
    
    coords = {
        'lat': target_lats,
        'lon': target_lons,
    }
    if has_time:
        coords['time'] = ds_utm.time.values
    
    ds_latlon = xr.Dataset(data_dict, coords=coords)
    
    # 添加属性
    ds_latlon.attrs['crs'] = 'EPSG:4326'
    ds_latlon.attrs['resolution_deg'] = resolution
    ds_latlon.attrs['source_utm_zone'] = utm_zone
    ds_latlon.attrs['center_lat'] = center_lat
    ds_latlon.attrs['center_lon'] = center_lon
    ds_latlon.lat.attrs['units'] = 'degrees_north'
    ds_latlon.lat.attrs['long_name'] = 'latitude'
    ds_latlon.lon.attrs['units'] = 'degrees_east'
    ds_latlon.lon.attrs['long_name'] = 'longitude'
    
    # 为降水变量添加单位
    for var in data_vars:
        if 'precip' in var.lower():
            ds_latlon[var].attrs['units'] = 'mm/h'
            ds_latlon[var].attrs['long_name'] = 'precipitation'
    
    return ds_latlon




def prediction_output_dataset(ds):
    """
    将 xarray 数据集从 1 小时分辨率上采样到 10 分钟分辨率。
    将 xarray 切片到目标区域。
    
    参数
    ----------
    ds : xarray.Dataset or xarray.DataArray
        具有逐小时分辨率的输入数据集
  
    返回
    -------
    xarray.Dataset or xarray.DataArray
        具有 10 分钟分辨率的上采样数据集
    """
    
    # 用于预测结果的 xr.dataset：
    ds = ds.isel(x=np.arange(96, 168+72)).isel(y=np.arange(96, 168+72)) 
    ds = ds.drop_vars(list(ds.data_vars)[1]).rename({list(ds.data_vars)[0]: "precipitation"})
    ds["precipitation"][:] = np.nan
    
    
    n_times_orig = len(ds.time)
    
    ds_upsampled = ds.resample(time='10min')
    ds_upsampled = ds_upsampled.ffill()
  
    
    # 重新采样不会填充最后一小时，因此需要手动扩展
    times_orig = pd.DatetimeIndex(ds.time.values)
    times_new = pd.date_range(
        start=times_orig[0],
        end=times_orig[-1] + pd.Timedelta(minutes=50),
        freq='10min'
    )

    # 重新索引以包含最后 50 分钟
    ds_upsampled = ds_upsampled.reindex(time=times_new, method='ffill')

    # 健全性检查
    n_times_new = len(ds_upsampled.time)
    expected_length = n_times_orig * 6
    
    if n_times_new != expected_length:
        raise ValueError(
            f"Length mismatch: new array has {n_times_new} steps, "
            f"expected {expected_length} (6 × {n_times_orig})"
        )
    
    return ds_upsampled
