"""
spateGAN-ERA5 模型推理引擎。

提供 InferenceEngine 类，用于通过滑动窗口推理运行模型预测。
"""

import numpy as np
import torch

class InferenceEngine:
    """用于运行 spateGAN 模型预测的推理引擎。
    
    处理降水降尺度中的模型加载、张量转换和滑动窗口推理。
    
    参数：
        model: 用于推理的 PyTorch 模型。
        sliding_step: 滑动窗口步长（默认：1）。
        device: 使用的 Torch 设备（如可用则自动检测 CUDA）。
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        sliding_step: int = 1,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.sliding_step = sliding_step
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def _to_tensor(self, array: np.ndarray | torch.Tensor) -> torch.Tensor:
        """将 numpy 数组或张量转换为指定设备上的张量。
        
        参数：
            array: 输入数组或张量。
            
        返回：
            位于配置设备上的张量。
        """
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array).float()
        return array.to(self.device)

    def infer(
        self,
        x: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor | None = None,
        seed: int = 1,
        return_numpy: bool = True,
    ) -> tuple[np.ndarray, ...] | np.ndarray | torch.Tensor:
        """
        沿时间维度使用滑动窗口运行推理。

        参数：
            x: 形状为 (B, C, T, H, W) 的输入，可为 torch.Tensor 或 np.ndarray
            target: 真值目标（可选）
            return_numpy: 是否以 numpy 返回预测（True）或以 torch 张量返回（False）

        返回：
            numpy 或 torch.Tensor 形式的预测结果
        """
        x = self._to_tensor(x)

        ## 后续可移动到独立的数据处理脚本
        # 将小于 0 的值截断为 0
        x = torch.clamp(x, min=0.0)
    
        # 确认输入中没有 NaN
        if torch.isnan(x).any():
            raise ValueError("Input contains NaNs. Please check the input data.")
        
        predictions = []
        T = x.shape[2]

        for i in range(4, T - 4, self.sliding_step):
            if i + 12 > T:
                continue

            first_slice = -1 * ((self.sliding_step * 6) - 48) // 2
            last_slice = ((self.sliding_step * 6) - 48) // 2

            if i - 4 == 0:
                first_slice = 0
            if i + 12 == T:
                last_slice = 48

            x_window = x[:, :, i - 4: i + 12]

            with torch.no_grad():
                pred = self.model(x_window, seed).cpu()

            pred = pred[:, :, first_slice:last_slice]
            predictions.append(pred)

        # 合并预测结果
        predictions = torch.cat(predictions, dim=2)

        # 裁剪目标和预测，移除扩展的 ERA5 时间信息（+-4 小时）和边界区域
        if target is not None:
            target = self._to_tensor(target)
            target = target[0, 0, 24:-24, 12:-12, 12:-12]
            target = target[6:-6].cpu()
            predictions = predictions[0, 0, 6:-6, 12:-12, 12:-12]
        else:
            predictions = predictions[0, 0]  # 移除 batch 和 channel 维度

        # 将输入数据裁剪到目标区域
        x = x.cpu()[0,:,5:-5, 8:-8, 8:-8]
        x = x.sum(dim=0)
        
       # 以 mm/h 返回预测和目标
        if return_numpy:
            return predictions.numpy()*6, target.numpy()*6, x.numpy() if target is not None else predictions.numpy()*6
        else:
            return predictions*6, target*6, x if target is not None else predictions*6
