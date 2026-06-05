"""
ERA5 降尺度流水线。

本模块提供主工作流函数，用于串联数据加载、预处理、投影和推理组件。
"""

import logging
from pathlib import Path

import xarray as xr

from dataloader import load_and_prepare_dataset
from downscaling_inference import ERA5DownscalingInference
from preprocessing import (
    calculate_domain_center,
    slice_data_for_projection,
    validate_patch_extraction,
    validate_time_dimension,
)
from projection import (
    latlon_to_utm,
    prediction_output_dataset,
    utm_to_latlon,
)
from utils import generate_output_filename

logger = logging.getLogger(__name__)


def load_era5_data(
    input_path: Path,
    precip_unit: str = "m",
    required_hours: int = 16,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[xr.Dataset, str]:
    """加载并准备 ERA5 数据集。
    
    参数：
        input_path: 输入 NetCDF 文件路径。
        precip_unit: 输入数据中的降水单位。
        required_hours: 最少所需时间步，单位为小时。
        start_date: 可选的筛选起始日期。
        end_date: 可选的筛选结束日期。
        
    返回：
        (准备后的数据集, 变量名) 元组。
        
    抛出：
        FileNotFoundError: 输入文件不存在时抛出。
        ValueError: 数据校验失败时抛出。
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info("Loading ERA5 data from %s", input_path)
    
    ds_era5, variable_name = load_and_prepare_dataset(
        str(input_path),
        precip_unit=precip_unit,
        precip_var=None,
        required_hours=required_hours,
        start_date=start_date,
        end_date=end_date,
    )
    
    logger.info("Loaded %d time steps: lat [%.1f, %.1f]°, lon [%.1f, %.1f]°",
                len(ds_era5.time),
                float(ds_era5.lat.min()),
                float(ds_era5.lat.max()),
                float(ds_era5.lon.min()),
                float(ds_era5.lon.max()))
    
    return ds_era5, variable_name


def extract_patch(
    ds: xr.Dataset,
    center_lat: float,
    center_lon: float,
    patch_size_km: float,
    patch_padding_km: float,
    extra_padding_cells: int,
    required_hours: int,
) -> tuple[xr.Dataset, dict, tuple[float, float]]:
    """校验区域并提取用于处理的 patch。
    
    参数：
        ds: 经纬度投影下的 ERA5 数据集。
        center_lat: patch 中心纬度。
        center_lon: patch 中心经度。
        patch_size_km: patch 大小，单位为千米。
        patch_padding_km: patch 周围填充，单位为千米。
        extra_padding_cells: 投影填充的额外单元数。
        required_hours: 最少所需数据小时数。
        
    返回：
        (切片数据集, 切片信息, (center_lat, center_lon)) 元组。
        
    抛出：
        ValueError: patch 提取校验失败时抛出。
    """
    logger.info("Validating patch extraction...")
    
    validate_patch_extraction(
        center_lat=center_lat,
        center_lon=center_lon,
        patch_size_km=672.0,
        patch_padding_km=100.0,
        lat_south=float(ds.lat.min().values),
        lat_north=float(ds.lat.max().values),
        lon_west=float(ds.lon.min().values),
        lon_east=float(ds.lon.max().values),
    )
    
    validate_time_dimension(ds, required_hours=required_hours)
    
    logger.info("Extracting patch centered at (%.4f°N, %.4f°E)...", center_lat, center_lon)
    
    ds_sliced, slicing_info = slice_data_for_projection(
        ds=ds,
        center_lat=center_lat,
        center_lon=center_lon,
        target_domain_size_km=672.0,
        extra_padding_cells=extra_padding_cells,
    )
    
    actual_center = calculate_domain_center(ds_sliced)
    
    return ds_sliced, slicing_info, actual_center


def project_to_utm(
    ds: xr.Dataset,
    center_lat: float,
    center_lon: float,
) -> tuple[xr.Dataset, xr.Dataset]:
    """将数据投影到两种分辨率的 UTM 坐标。
    
    参数：
        ds: 经纬度投影下的数据集。
        center_lat: 用于确定 UTM 分区的中心纬度。
        center_lon: 用于确定 UTM 分区的中心经度。
        
    返回：
        (高分辨率 336x336 数据集, 低分辨率 28x28 数据集) 元组。
    """
    ds_utm_336 = latlon_to_utm(
        ds, center_lat=center_lat, center_lon=center_lon,
        target_size=336, grid_spacing=2000, method="nearest",
    )
    
    ds_utm_28 = latlon_to_utm(
        ds, center_lat=center_lat, center_lon=center_lon,
        target_size=28, grid_spacing=24000, method="nearest",
    )
    
    return ds_utm_336, ds_utm_28


def run_inference(
    ds_utm_28: xr.Dataset,
    ds_utm_336: xr.Dataset,
    config: dict,
) -> xr.Dataset:
    """在准备好的数据上运行模型推理。
    
    参数：
        ds_utm_28: 低分辨率 UTM 输入数据集。
        ds_utm_336: 用作输出模板的高分辨率 UTM 数据集。
        config: 配置字典。
        
    返回：
        UTM 投影下的预测数据集
    """
    ds_utm_pred = prediction_output_dataset(ds_utm_336)
    
    inference = ERA5DownscalingInference(
        config=config,
        device=config["processing"]["device"],
        seed=config["processing"]["seed"],
    )
    
    x_tensor = inference.prepare_tensor_dataset(
        ds_utm_28, variable_names=list(ds_utm_28.data_vars)
    )
        
    logger.info("Running inference...")
    predictions = inference.predict_sliding_window(
        x_tensor,
        ds_prediction=ds_utm_pred,
        slide=config["inference"]["stride_hours"],
    )
    
    return predictions


def save_outputs(
    predictions_utm: xr.Dataset,
    ds_utm_28: xr.Dataset,
    output_utm_dir: Path | None,
    output_latlon_dir: Path | None,
    config: dict,
    save_model_input: bool = False,
) -> None:
    """将预测输出保存到文件。
    
    参数：
        predictions_utm: UTM 投影下的预测结果。
        ds_utm_28: UTM 投影下的模型输入。
        output_utm_dir: UTM 输出目录。
        output_latlon_dir: 经纬度输出目录。
        config: 配置字典。
        save_model_input: 是否随预测结果一起保存模型输入。
    """
    if output_latlon_dir:
        output_latlon_dir.mkdir(parents=True, exist_ok=True)
        
        predictions_latlon = utm_to_latlon(predictions_utm, resolution=0.018)
        
        filename = generate_output_filename(predictions_latlon, config, projection="latlon")
        predictions_latlon.to_netcdf(output_latlon_dir / filename)
        logger.info("Saved: %s", output_latlon_dir / filename)
        
        if save_model_input:
            ds_latlon = utm_to_latlon(ds_utm_28, resolution=0.25)
            filename_era = generate_output_filename(ds_latlon, config, projection="latlon", model="era5")
            ds_latlon.to_netcdf(output_latlon_dir / filename_era)
            logger.info("Saved: %s", output_latlon_dir / filename_era)
    
    if output_utm_dir:
        output_utm_dir.mkdir(parents=True, exist_ok=True)
        
        filename = generate_output_filename(predictions_utm, config, projection="utm")
        predictions_utm.to_netcdf(output_utm_dir / filename)
        logger.info("Saved: %s", output_utm_dir / filename)
        
        if save_model_input:
            filename_era = generate_output_filename(ds_utm_28, config, projection="utm", model="era5")
            ds_utm_28.to_netcdf(output_utm_dir / filename_era)
            logger.info("Saved: %s", output_utm_dir / filename_era)


def _create_precipitation_sums_plot(
    predictions_utm: xr.Dataset,
    ds_utm_28: xr.Dataset,
    project_root: Path,
    config: dict,
) -> None:
    """创建一个简单的降水总量并排对比图。

    左图：(ds_utm_28.cp + ds_utm_28.lsp).sum(dim='time')
    右图：predictions_utm.precipitation.sum(dim='time')

    将 PNG 保存到 `project_root / plots`（或 config['data']['plots_path']）。
    """
    plotting_cfg = config.get("plotting", {})
    if not plotting_cfg.get("precipitation_sums", False):
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        left = None
        if ("cp" in ds_utm_28.data_vars) and ("lsp" in ds_utm_28.data_vars):
            left = (ds_utm_28["cp"] + ds_utm_28["lsp"]).sum(dim="time")
            left = left[8:-8, 8:-8]
        else:
            logger.warning("ds_utm_28 missing 'cp' and/or 'lsp' variables; left plot will be empty")

        right = None
        if "precipitation" in predictions_utm.data_vars:
            right = (predictions_utm["precipitation"]/6).sum(dim="time") # 除以 6，因为每小时有 6 个时间步
        else:
            logger.warning("predictions_utm missing 'precipitation' variable; right plot will be empty")

        if (left is None) and (right is None):
            logger.warning("No data available for precipitation sums plotting; skipping")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if left is not None:
            left.plot(ax=axes[0], cmap="turbo")
            axes[0].set_title("Input (target domain) precipitation sum (cp + lsp) - mm")
        else:
            axes[0].axis("off")

        if right is not None:
            right.plot(ax=axes[1], cmap="turbo")
            axes[1].set_title("Predicted precipitation sum - mm")
        else:
            axes[1].axis("off")

        plt.tight_layout()

        plots_dir = (
            project_root / config.get("data", {}).get("plots_path", "plots")
        )
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_name = generate_output_filename(predictions_utm, config, projection="utm")
        plot_name = "precip_sums_" + plot_name.replace(".nc", ".png")
        out_path = plots_dir / plot_name
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        logger.info("Saved precipitation sums plot to %s", out_path)

    except Exception as e:
        logger.exception("Error creating precipitation sums plot: %s", e)


def fill_nans_if_sparse(ds: xr.Dataset, threshold=0.01) -> xr.Dataset:
    # 所有变量中的总值数量
    total_values = sum(da.size for da in ds.data_vars.values())

    # 所有变量中的 NaN 总数
    nan_values = sum(
        int(da.isnull().sum().values)
        for da in ds.data_vars.values()
    )

    nan_fraction = nan_values / total_values

    if nan_fraction < threshold and nan_fraction > 0:
        logger.info(f"Filling {nan_fraction:.4f} NaNs (below threshold {threshold:.4f})")
        return ds.fillna(0)
    elif nan_fraction >= threshold:
        logger.info(f"Not filling Nans since above threshold of {threshold:.4f}")
        assert nan_fraction < threshold, "Dataset too sparse with NaNs"
    else:
        return ds


def run_downscaling_pipeline(config: dict, project_root: Path) -> None:
    """运行完整的降尺度流水线。
    
    这是主编排函数，串联全部流水线阶段：
    加载 -> 提取 -> 投影 -> 推理 -> 保存。
    
    参数：
        config: 配置字典。
        project_root: 项目根目录。
        
    抛出：
        FileNotFoundError: 输入文件不存在时抛出。
        ValueError: 校验失败时抛出。
    """
    # 阶段 1：加载数据
    input_path = project_root / config["data"]["input_path"]
    ds_era5, _ = load_era5_data(
        input_path=input_path,
        precip_unit=config["data"].get("precip_unit", "m"),
        required_hours=16,
        start_date=config["time"].get("start_date"),
        end_date=config["time"].get("end_date"),
    )
    
    # 阶段 2：提取 patch
    ds_sliced, _, center_coords = extract_patch(
        ds=ds_era5,
        center_lat=config["domain"]["center_lat"],
        center_lon=config["domain"]["center_lon"],
        patch_size_km=672.0,
        patch_padding_km=100.0,
        extra_padding_cells=config["projection"]["extra_padding_cells"],
        required_hours=16,
    )
    
    logger.info("running UTM projection")
    
    # 阶段 3：投影到 UTM
    ds_utm_336, ds_utm_28 = project_to_utm(ds_sliced, *center_coords)
    
    # 阶段 3.1：如果 NaN 很稀疏，则填充 NaN
    ds_utm_28 = fill_nans_if_sparse(ds_utm_28, threshold=config.get("data", {}).get("nan_fill_threshold", 0.01))
        
    # 阶段 4：运行推理
    predictions_utm = run_inference(ds_utm_28, ds_utm_336, config)
    
    # 阶段 4：绘制降水图（可选）
    
    # 阶段 5：保存输出
    output_utm_dir = (
        project_root / config["data"]["output_utm_path"]
        if config["data"].get("output_utm_path") else None
    )
    output_latlon_dir = (
        project_root / config["data"]["output_latlon_path"]
        if config["data"].get("output_latlon_path") else None
    )
    
    save_outputs(
        predictions_utm=predictions_utm,
        ds_utm_28=ds_utm_28,
        output_utm_dir=output_utm_dir,
        output_latlon_dir=output_latlon_dir,
        config=config,
        save_model_input=config["data"].get("save_model_input", False),
    )

    # 可选的简单绘图
    try:
        _create_precipitation_sums_plot(predictions_utm, ds_utm_28, project_root, config)
    except Exception:
        # _create_precipitation_sums_plot 会自行记录错误
        pass
    
    logger.info("Downscaling pipeline completed successfully!")
