"""spateGAN-ERA5 的神经网络模型定义。包含用于降水降尺度的 Generator 模型及其辅助模块。
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class CustomDropout(nn.Module):
    """在时间维度上使用相同掩码的自定义 dropout。
    这可确保时间序列数据具有一致的 dropout 模式。
    参数：
        p: Dropout 概率（0-1）。
        d_seed: 用于可复现性的随机种子。
    """
    
    def __init__(self, p: float, d_seed: int) -> None:
        super().__init__()
        self.p = p
        torch.manual_seed(d_seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用时间一致的 dropout 掩码。
        
        参数：
            x: 形状为 (batch, channels, time, height, width) 的输入张量。
            
        返回：
            应用 dropout 后的张量。
        """
        device = x.device
        batch, channels, time, height, width = x.shape

        mask_shape = (batch, channels, 1, height, width)
        mask = torch.bernoulli(torch.ones(mask_shape, device=device) * (1 - self.p))
        mask = mask.repeat(1, 1, time, 1, 1) / (1 - self.p)
        
        return x * mask


class ResidualBlock3D(nn.Module):
    """可选实例归一化的 3D 残差块。
    参数：
        in_channels: 输入通道数。
        out_channels: 输出通道数。
        use_layer_norm: 是否使用实例归一化。
        stride: 卷积步长。
        padding_type: 为 True 时使用反射填充。
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_layer_norm: bool = True,
        stride: int = 1,
        padding_type: bool | None = None,
    ) -> None:
        super().__init__()

        padding = 0 if padding_type else 1
        self.use_layer_norm = use_layer_norm
        self.padding_type = padding_type

        self.padding_layer = nn.ReflectionPad3d(1) if padding_type else None

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=padding, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=False)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=padding, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=False)

        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.adjust_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1,
                                         stride=stride, bias=False)
            self.adjust_norm = nn.InstanceNorm3d(out_channels)
        else:
            self.adjust_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """残差块的前向传播。
        
        参数：
            x: 形状为 (batch, channels, time, height, width) 的输入张量。
            
        返回：
            应用残差连接后的输出张量。
        """
        residual = x

        if self.padding_layer:
            x = self.padding_layer(x)

        out = self.conv1(x)
        if self.use_layer_norm:
            out = self.norm1(out)
        out = self.relu(out)

        if self.padding_layer:
            out = self.padding_layer(out)

        out = self.conv2(out)
        if self.use_layer_norm:
            out = self.norm2(out)

        if self.adjust_conv:
            residual = self.adjust_conv(residual)
            residual = self.adjust_norm(residual)

        out += residual
        return self.relu(out)


class Interpolate(nn.Module):
    """用于上采样的三线性插值模块。
    参数：
        scale_factor: 各维度（time, height, width）的缩放因子。
        mode: 插值模式（默认：'trilinear'）。
    """
    
    def __init__(self, scale_factor: tuple[int, int, int], mode: str = 'trilinear') -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入张量进行上采样。
        
        参数：
            x: 形状为 (batch, channels, time, height, width) 的输入张量。
            
        返回：
            上采样后的张量。
        """
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)


class Constraint(nn.Module):
    """用于强制 ERA5 降水守恒的约束层。
    缩放预测结果，使其匹配 ERA5 总降水量。
    """
    
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, prediction: torch.Tensor, constraint: torch.Tensor
    ) -> torch.Tensor:
        """应用约束以匹配 ERA5 总量。
        
        参数：
            prediction: 模型预测张量。
            constraint: ERA5 约束张量。
            
        返回：
            缩放后的预测张量。
        """
        constraint = constraint[:, :, 5:-5, 8:-8, 8:-8].sum(dim=1, keepdim=True)
        scale = (constraint[:, 0].mean(dim=(1, 2, 3)) / 6).view(-1, 1, 1, 1, 1)
        pred_mean = prediction[:, 0].mean(dim=(1, 2, 3)).view(-1, 1, 1, 1, 1)
        return prediction * (scale / pred_mean)


class Generator(nn.Module):
    """用于降水降尺度的 spateGAN 生成器。
    
    使用残差卷积架构，将粗分辨率 ERA5 降水数据转换为高分辨率降水场。
    
    输入形状：(batch, 2, 16, 28, 28) - 2 个通道（CP, LSP）、16 小时、28x28 网格
    输出形状：(batch, 1, 48, 168, 168) - 10 分钟分辨率的 8 小时数据、168x168 网格
    """
    def __init__(self):
        super().__init__()

        self.filter_size = 96
        self._initialize_layers()

    def _initialize_layers(self):
        """初始化模型层。"""
        f = self.filter_size

        self.input_pad = nn.ReflectionPad3d((1, 1, 1, 1, 0, 0))

        self.res1 = ResidualBlock3D(2, f, use_layer_norm=False, padding_type=True)
        self.res2 = ResidualBlock3D(f, f, use_layer_norm=False, padding_type=True)
        self.res3 = ResidualBlock3D(f, f, use_layer_norm=True, padding_type=True)

        self.down0 = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(f, f, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )

        self.up0 = Interpolate((2, 2, 2))
        self.res4 = ResidualBlock3D(f, f, padding_type=True)

        self.up1 = Interpolate((1, 2, 2))
        self.res5 = ResidualBlock3D(f, f, padding_type=True)

        self.up2 = Interpolate((3, 1, 1))
        self.res6 = ResidualBlock3D(f, f, padding_type=True)

        self.up3 = Interpolate((1, 3, 3))
        self.res7 = ResidualBlock3D(f, f, padding_type=True)

        self.res8 = ResidualBlock3D(f, f, padding_type=True)
        self.res9 = ResidualBlock3D(f, f, use_layer_norm=False, padding_type=True)

        self.output_conv = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(f, 1, kernel_size=3, padding=0),
            nn.Softplus()
        )

        self.constraint_layer = Constraint()

    def forward(self, x: torch.Tensor, dropout_seed: int) -> torch.Tensor:
        """从 ERA5 输入生成高分辨率降水。
        
        参数：
            x: 形状为 (batch, 2, 16, height, width) 的输入张量。
            dropout_seed: 用于保持 dropout 一致性的随机种子。
            
        返回：
            高分辨率降水张量。
        """
        x1 = self.res1(x)
        x1 = CustomDropout(p=0.2, d_seed=dropout_seed)(x1)
        x2_stay = self.res2(x1)

        x2 = self.down0(x2_stay)
        
        x2 = x2_stay[:, :, 4:-4, 7:-7, 7:-7] + x2
        x2 = self.res3(x2)
        x2 = CustomDropout(p=0.2, d_seed=dropout_seed)(x2)

        x2 = self.up0(x2)
        x2 = self.res4(x2)

        x2 = self.up1(x2)
        x2 = self.res5(x2)
        x2 = CustomDropout(p=0.2, d_seed=dropout_seed)(x2)

        x2 = self.up2(x2)
        x2 = self.res6(x2)

        x2 = self.up3(x2)
        x2 = self.res7(x2)

        x2 = self.res8(x2)
        x2 = self.res9(x2)

        output = self.output_conv(x2)

        output[:,:,6:-6,12:-12,12:-12] = self.constraint_layer(output[:,:,6:-6,12:-12,12:-12], x)
        
        return output


class DiscriminatorResidualBlock3D(nn.Module):
    """3D residual block with zero padding for the discriminator."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int, int] = 1,
        use_instance_norm: bool = True,
    ) -> None:
        super().__init__()
        self.use_instance_norm = use_instance_norm

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=False)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=False)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.adjust = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels, affine=False),
            )
        else:
            self.adjust = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.adjust is None else self.adjust(x)

        out = self.conv1(x)
        if self.use_instance_norm:
            out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_instance_norm:
            out = self.norm2(out)

        return self.relu(out + residual)


class MultiplicativeGaussianNoise(nn.Module):
    """Multiplicative input noise with mean 1 and std 0.05."""

    def __init__(self, mean: float = 1.0, std: float = 0.05) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.std == 0:
            return x
        noise = torch.normal(
            mean=self.mean,
            std=self.std,
            size=x.shape,
            device=x.device,
            dtype=x.dtype,
        )
        return x * noise


class Discriminator(nn.Module):
    """Spatio-temporal cGAN discriminator for training reproduction.

    Inputs:
        x: ERA5 context, shape ``(B, 2, 16, 28, 28)``.
        y: high-resolution observation or prediction, shape ``(B, 1, 48, 168, 168)``.

    Output:
        Logits with shape ``(B, 1)``. Use ``BCEWithLogitsLoss``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.noise_x = MultiplicativeGaussianNoise()
        self.noise_y = MultiplicativeGaussianNoise()

        self.high_path = nn.Sequential(
            DiscriminatorResidualBlock3D(1, 128, stride=(2, 2, 2), use_instance_norm=False),
            DiscriminatorResidualBlock3D(128, 128, stride=(3, 2, 2), use_instance_norm=True),
            DiscriminatorResidualBlock3D(128, 128, stride=(1, 3, 3), use_instance_norm=True),
        )

        self.low_path = nn.Sequential(
            DiscriminatorResidualBlock3D(2, 64, stride=(2, 2, 2), use_instance_norm=False),
            DiscriminatorResidualBlock3D(64, 32, stride=1, use_instance_norm=False),
        )

        self.fusion = DiscriminatorResidualBlock3D(
            128 + 32,
            64,
            stride=(2, 2, 2),
            use_instance_norm=True,
        )
        self.output_conv = nn.Conv3d(64, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or y.ndim != 5:
            raise ValueError("Expected x and y to be 5D tensors: (B, C, T, H, W)")

        high = self.high_path(self.noise_y(y))
        low = self.low_path(self.noise_x(x))

        if high.shape[2:] != low.shape[2:]:
            raise ValueError(
                f"Feature shapes do not align: high={tuple(high.shape)}, low={tuple(low.shape)}"
            )

        fused = torch.cat([high, low], dim=1)
        out = self.fusion(fused)
        out = self.output_conv(out)
        return out.mean(dim=(2, 3, 4))


class CNNDownscaler(nn.Module):
    """简单 3D CNN 降尺度基线。

    输入:  (B, 2, 16, 28, 28)
    输出:  (B, 1, 48, 168, 168)
    """

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Upsample(
            scale_factor=(3, 6, 6),
            mode="trilinear",
            align_corners=False,
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.upsample(x)
        return self.decoder(x)
