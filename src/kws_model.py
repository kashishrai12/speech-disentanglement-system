# kws_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TCBlock(nn.Module):
    """
    Temporal Convolution block with residual connection.
    Depthwise separable convolution reduces parameter count dramatically.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Sequential(
            # Depthwise conv: each channel convolved independently
            nn.Conv1d(in_channels, in_channels, kernel_size,
                      stride=stride, padding=padding, groups=in_channels, bias=False),
            # Pointwise conv: mix channels
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=padding, groups=out_channels, bias=False),
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        # Residual connection — match dimensions if needed
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
        ) if in_channels != out_channels or stride != 1 else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.residual(x))


class MultiHeadAttentionPooling(nn.Module):
    """
    Attention pooling over time — the model learns WHICH frames to focus on,
    rather than just averaging all frames. Critical for variable-length words.
    """
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        # x: [B, C, T] → [B, T, C]
        x = x.permute(0, 2, 1)
        q = self.query.expand(x.size(0), -1, -1)  # [B, 1, C]
        out, _ = self.attention(q, x, x)            # [B, 1, C]
        return out.squeeze(1)                        # [B, C]


class TCResNetKWS(nn.Module):
    """
    TC-ResNet for Keyword Spotting.
    
    Architecture:
        Input:   [B, 1, 40, T]  (log-Mel: batch, channels, mels, time)
        Reshape: [B, 40, T]     (treat Mel bins as channels)
        TC-ResNet blocks with increasing channels
        Multi-head attention pooling
        Classifier head
    
    Target: ~600K parameters (leaves room for speaker verifier)
    """
    def __init__(
        self,
        n_mels: int = 40,
        num_classes: int = 2,   # [other, target_word]
        channels: list = [40, 64, 96, 128, 128],
    ):
        super().__init__()

        # Initial projection
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # TC-ResNet blocks
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            stride = 2 if i % 2 == 0 else 1   # downsample every other block
            self.blocks.append(TCBlock(channels[i], channels[i+1],
                                       kernel_size=9, stride=stride))

        # Attention pooling: captures the most informative temporal positions
        self.pool = MultiHeadAttentionPooling(channels[-1], num_heads=4)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 40, T] → [B, 40, T]
        x = x.squeeze(1)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)           # [B, 128]
        return self.classifier(x)  # [B, 2]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)