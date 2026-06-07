"""
用于灵活 ERA5 降尺度的推理封装模块。

负责模型加载、数据准备和推理执行。
"""

from pathlib import Path

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

from .model import Generator


class ERA5DownscalingInference:
    """用于 ERA5 数据上 spateGAN 模型推理的封装类。
    
    处理高分辨率降水降尺度中的模型加载、张量准备和滑动窗口预测。
    
    参数：
        config: 包含模型和处理设置的配置字典。
        device: 计算设备（'cuda' 或 'cpu'）。
        seed: 用于可复现性的随机种子。
    """
    
    def __init__(
        self,
        config: dict,
        device: str = 'cuda',
        seed: int = 42,
    ) -> None:
        """
        初始化推理引擎。
        
        参数：
            model_weights_path: 模型权重文件路径
            device: 'cuda' 或 'cpu'
            seed: 用于可复现性的随机种子
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        model_weights_path = config['data']['model_weights_path']
        self.final_era_constraint = config['inference'].get('final_era_constraint', True)        
        
        # 加载模型
        self.model = Generator().to(self.device)
        checkpoint = torch.load(model_weights_path, weights_only=True, map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()
    
    def prepare_tensor_dataset(
        self,
        dataset: xr.Dataset,
        variable_names: list[str] | None = None,
    ) -> torch.Tensor:
        """
        从 xarray Dataset 准备张量数据集。
        
        参数：
            dataset: 包含变量的输入 xarray Dataset
            variable_names: 要提取的变量名列表。为 None 时使用所有数据变量
        
        返回：
            形状为 (1, channels, time, height, width) 的 torch.Tensor
        
        抛出：
            ValueError: 请求的变量在数据集中不存在时抛出
        """
        if variable_names is None:
            variable_names = list(dataset.data_vars)
        
        data_arrays: list[np.ndarray] = []
        
        for var_name in variable_names:
            if var_name not in dataset.data_vars:
                raise ValueError(f"Variable '{var_name}' not found in dataset")
            
            data = dataset[var_name].values  # (time, y, x)
            data_arrays.append(data)
        
        # 堆叠变量：list of (time, y, x) -> (channels, time, y, x)
        stacked_data = np.stack(data_arrays, axis=0)
        
        # 添加 batch 维度：(channels, time, y, x) -> (1, channels, time, y, x)
        batched_data = np.expand_dims(stacked_data, axis=0)
        
        # 转换为张量并移动到计算设备
        tensor = torch.from_numpy(batched_data).float().to(self.device)
        
        return tensor
    
    def predict_sliding_window(
        self,
        x: torch.Tensor,
        ds_prediction: xr.Dataset | None = None,
        slide: int = 8,
    ) -> np.ndarray | xr.Dataset:
        """
        使用滑动窗口应用模型并拼接预测结果。
        
        参数：
            x: 形状为 (batch, channels, time, height, width) 的输入张量，time >= 16
            slide: 滑动窗口步长，单位为小时（1-8）
        
        返回：
            形状为 (time * 6, height, width) 的 numpy.ndarray，包含拼接后的预测。
            其中 6 来自 10 分钟分辨率（每小时 6 个步长）。
        
        抛出：
            ValueError: 输入少于 16 个时间步或 slide 超出范围时抛出
        """
        n_times = x.shape[2]
        h, w = 144, 144
        
        if n_times < 16:
            raise ValueError(f"Input must have at least 16 timesteps, got {n_times}")
        
        if not 1 <= slide <= 8:
            raise ValueError(f"Slide must be between 1 and 8, got {slide}")
        
        steps_per_hour = 6  # 10 分钟分辨率
        steps_to_keep = slide * steps_per_hour
        
        # 计算中间窗口的中心片段索引
        center_start = (48 - steps_to_keep) // 2
        center_end = center_start + steps_to_keep
        
        # 初始化预测列表，并用 NaN 填充前 4 小时
        predictions: list[np.ndarray] = [
            np.full((4 * steps_per_hour, h, w), np.nan)
        ]
                
        # 生成滑动窗口位置
        positions = list(range(0, n_times - 15, slide))
        
        for i, pos in enumerate(tqdm(positions, desc='Downscaling process', dynamic_ncols=True)):
            x_window = x[:, :, pos:pos + 16]
            
            with torch.no_grad():
                if self.seed == -1:
                    seed = torch.randint(0, 10000, (1,))
                else:
                    seed = self.seed
                pred = self.model(x_window, seed)  # 形状：(batch, channels, 48, h, w)
                
            
            # 提取预测，移除 batch 和 channel 维度，并裁剪边界
            pred = pred.detach().cpu().numpy()[0, 0, :, 12:-12, 12:-12]
            
            is_first = (i == 0)
            is_last = (i == len(positions) - 1)
            
            if is_first and is_last:
                # 只有一个窗口：保留全部
                predictions.append(pred)
            elif is_first:
                # 第一个窗口：保留从开头到 center_end 的部分
                predictions.append(pred[:center_end])
            elif is_last:
                # 最后一个窗口：保留从 center_start 到结尾的部分
                predictions.append(pred[center_start:])
            else:
                # 中间窗口：只保留中心部分
                predictions.append(pred[center_start:center_end])
        
        # 拼接所有预测结果
        result = np.concatenate(predictions, axis=0)
        
        # 如有必要，用 NaN 填充到期望输出长度
        expected_length = n_times * steps_per_hour
        current_length = result.shape[0]
        
        if current_length < expected_length:
            end_padding = np.full((expected_length - current_length, h, w), np.nan)
            result = np.concatenate([result, end_padding], axis=0)
            
            
        if self.final_era_constraint:
            # 应用最终 ERA5 约束
            constraint = x[0,:, 4:-4, 8:-8, 8:-8].sum(dim=0, keepdim=False).detach().cpu().numpy()
            scale = constraint.mean() / 6
            pred_mean = np.nanmean(result)
            result = result * (scale / pred_mean)
            
            
        if ds_prediction is not None:
            if result.shape[0] != len(ds_prediction.time):
                raise ValueError(f"Prediction length {result.shape[0]} does not match ds_prediction time length {len(ds_prediction.time)}")
            else:
                ds_output = ds_prediction.copy(deep=True)
                # 保持空间维度与 UTM 数据集一致（y 为北向坐标，x 为东向坐标）
                ds_output['precipitation'] = (['time', 'y', 'x'], result*6)
                ds_output['precipitation'].attrs['units'] = 'mm/h'
                ds_output.attrs['time zone'] = 'UTC'
                return ds_output # 返回 mm/h 单位的降水预测
        else:
            return result * 6 # 返回 mm/h 单位的降水预测
    
