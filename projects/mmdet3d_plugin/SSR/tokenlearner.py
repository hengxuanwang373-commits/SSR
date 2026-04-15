import torch
import torch.nn as nn
import torch.nn.functional as F
import math
"from https://github.com/Kashu7100/TokenLearner/blob/main/model.py"


# class ChannelAttention(nn.Module):
#     """CBAM Channel Attention Module.
#     Uses both max-pool and avg-pool, followed by a shared MLP.
#     """
#     def __init__(self, channels, reduction=4):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.max_pool = nn.AdaptiveMaxPool1d(1)
#         self.mlp = nn.Sequential(
#             nn.Linear(channels, channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x: [B, C, H, W] -> reshape to [B, C, H*W]
#         if x.dim() == 4:
#             b, c, h, w = x.shape
#             x_flat = x.view(b, c, h * w)
#         else:
#             x_flat = x
#         avg_out = self.mlp(self.avg_pool(x_flat).squeeze(-1))
#         max_out = self.mlp(self.max_pool(x_flat).squeeze(-1))
#         out = self.sigmoid(avg_out + max_out).unsqueeze(-1)
#         if x.dim() == 4:
#             return x * out.view(b, c, 1, 1)
#         return x * out
#
#
# class SpatialAttention(nn.Module):
#     """CBAM Spatial Attention Module.
#     Uses pooling along channel dimension, then a convolution.
#     """
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x: [B, C, H, W]
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avg_out, max_out], dim=1)
#         out = self.sigmoid(self.conv(out))
#         return x * out
#
#
# class CBAM(nn.Module):
#     """CBAM: Convolutional Block Attention Module.
#     Combines channel attention and spatial attention sequentially.
#
#     Args:
#         channels (int): Number of input channels (C).
#         reduction (int): Channel attention reduction ratio. Default: 4.
#         kernel_size (int): Spatial attention kernel size. Default: 7.
#     """
#     def __init__(self, channels, reduction=4, kernel_size=7):
#         super().__init__()
#         self.channel_attention = ChannelAttention(channels, reduction)
#         self.spatial_attention = SpatialAttention(kernel_size)
#
#     def forward(self, x):
#         """
#         Args:
#             x: Input tensor with shape [B, C, H, W]
#         Returns:
#             Tensor with shape [B, C, H, W]
#         """
#         x = self.channel_attention(x)
#         x = self.spatial_attention(x)
#         return x

class MlpBlock(nn.Module):
    """Simple MLP block with GELU activation and dropout."""
    def __init__(self, input_dim, mlp_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class TokenLearnerV11(nn.Module):
    """TokenLearner module Version 1.1 for PyTorch."""
    def __init__(self, num_tokens, in_channels, bottleneck_dim=64, dropout_rate=0.):
        super(TokenLearnerV11, self).__init__()
        self.num_tokens = num_tokens
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate
        self.layer_norm = nn.GroupNorm(1, in_channels, eps=1e-6)
        self.mlp = MlpBlock(input_dim=in_channels, mlp_dim=self.bottleneck_dim, output_dim=self.num_tokens, dropout_rate=self.dropout_rate)

    def forward(self, inputs, deterministic=True):
        """
        Args: 
            inputs: Inputs of shape `[B, HW, C]` or `[B, C, H, W]`.
            
        Returns:
            [B, num_token, C]
        """
        if inputs.dim() == 4:
            n, c, h, w = inputs.shape
            inputs = inputs.view(n, c, h * w).permute(0,2,1)

        selected = self.mlp(self.layer_norm(inputs.permute(0,2,1)).permute(0,2,1))

        # Softmax normalization
        # selected [B, num_token, HW]
        selected = selected.view(inputs.shape[0], self.num_tokens, -1).softmax(dim=-1)

        # Weighted sum based on the selected tokens
        # feat [B, HW, C]
        feat = inputs.view(inputs.shape[0], -1, inputs.shape[-1])
        outputs = torch.einsum('bsi,bic->bsc', selected, feat)
        # outputs [B, num_token, C]
        return outputs, selected
    
    
class TokenFuser(nn.Module):
    """Token fusion module in PyTorch."""
    def __init__(self, num_tokens, in_channels, use_normalization=True, bottleneck_dim=64, dropout_rate=0.):
        super().__init__()
        self.use_normalization = use_normalization
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate
        self.norm = nn.GroupNorm(1, in_channels, eps=1e-6)
        self.norm1 = nn.GroupNorm(1, in_channels, eps=1e-6) if use_normalization else None
        self.norm2 = nn.GroupNorm(1, in_channels, eps=1e-6) if use_normalization else None
        self.dense = nn.Linear(in_features=num_tokens, out_features=bottleneck_dim)
        self.mlp = MlpBlock(input_dim=in_channels, mlp_dim=bottleneck_dim, output_dim=bottleneck_dim, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, original):
        """Applies token fusion to the generate 2D ouputs.

        Args:
            inputs: Inputs of shape `[B, n_token, C]`.
            original: Inputs of shape `[B, HW, C]` or `[B, C, H, W]`.

        Returns:
            Output tensor with the shape identical to `original'.
        """
        is_2d = False
        if original.dim() == 4:
            is_2d = True
            n, c, h, w = original.shape
            original = original.view(n, c, h * w).permute(0,2,1)

        if self.use_normalization:
            inputs = self.norm1(inputs.permute(0,2,1)).permute(0,2,1)

        # inputs [B, C, D]
        inputs = self.dense(inputs.permute(0,2,1))
        
        if self.use_normalization:
            inputs = self.norm2(inputs)

        original = self.norm(original.permute(0,2,1)).permute(0,2,1)
        # mix [B, HW, D]
        mix = self.mlp(original).sigmoid()
    
        # Using matrix multiplication for fusing tokens back
        fused = torch.einsum('bcs,bhs->bhc', inputs, mix)
        fused = self.dropout(fused)

        if is_2d:
            fused = fused.view(n, h, w, c).permute(0,3,1,2)

        return fused
