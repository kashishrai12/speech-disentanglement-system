# speaker_verifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: reweights channels by global context."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).unsqueeze(-1)
        return x * w


class Res2Block(nn.Module):
    """
    Res2Net-style block: hierarchical multi-scale feature extraction.
    The key innovation: splits channels into K groups, processes them
    in a chain (each group sees the previous group's output + its own input).
    This gives a large effective receptive field without large kernels.
    """
    def __init__(self, channels: int, scale: int = 8, dilation: int = 1):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale

        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 3,
                      padding=dilation, dilation=dilation, bias=False)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale - 1)])
        self.se = SEBlock(channels)
        self.bn_out = nn.BatchNorm1d(channels)

    def forward(self, x):
        chunks = torch.chunk(x, self.scale, dim=1)
        outs = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                outs.append(chunk)
            elif i == 1:
                outs.append(F.relu(self.bns[i-1](self.convs[i-1](chunk))))
            else:
                outs.append(F.relu(self.bns[i-1](self.convs[i-1](chunk + outs[-1]))))
        out = torch.cat(outs, dim=1)
        out = self.se(out)
        return F.relu(self.bn_out(out + x))


class AttentiveStatisticsPooling(nn.Module):
    """
    Computes mean and std of frame-level features, weighted by learned attention.
    Outputs a fixed-size embedding regardless of input length.
    This is the core of TDNN-based speaker embedding.
    """
    def __init__(self, in_dim: int, bottleneck: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim * 3, bottleneck, 1),
            nn.Tanh(),
            nn.Conv1d(bottleneck, in_dim, 1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        # x: [B, C, T]
        global_mean = x.mean(dim=2, keepdim=True).expand_as(x)
        global_std  = x.std(dim=2, keepdim=True).clamp(min=1e-8).expand_as(x)
        attn_input = torch.cat([x, global_mean, global_std], dim=1)
        weights = self.attention(attn_input)           # [B, C, T]
        mean = (weights * x).sum(dim=2)                # [B, C]
        std  = (weights * (x - mean.unsqueeze(2))**2).sum(dim=2).clamp(min=1e-8).sqrt()
        return torch.cat([mean, std], dim=1)           # [B, 2C]


class MiniECAPATDNN(nn.Module):
    """
    Lightweight (distilled) ECAPA-TDNN for speaker verification.
    
    Full ECAPA-TDNN: ~6M params, embedding_dim=512
    This version: ~1.5M params, embedding_dim=192
    
    Training objective: ArcFace loss (angular margin softmax)
    At inference: cosine similarity between enrollment and test embeddings
    """
    def __init__(self, in_channels: int = 40, embedding_dim: int = 192, num_speakers: int = None):
        super().__init__()

        C = 384   # channel width (reduced from 512 in original)

        # Input projection
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, C, 5, padding=2, bias=False),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True),
        )

        # Res2 blocks with increasing dilation
        self.layer1 = Res2Block(C, scale=4, dilation=2)
        self.layer2 = Res2Block(C, scale=4, dilation=3)
        self.layer3 = Res2Block(C, scale=4, dilation=4)

        # Aggregation: multi-scale feature fusion
        self.aggregate = nn.Sequential(
            nn.Conv1d(C * 3, C, 1, bias=False),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True),
        )

        # Attentive statistics pooling
        self.asp = AttentiveStatisticsPooling(C, bottleneck=128)

        # Bottleneck to embedding
        self.embedding = nn.Sequential(
            nn.Linear(C * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # Classification head for training (ArcFace)
        if num_speakers:
            self.classifier = nn.Linear(embedding_dim, num_speakers, bias=False)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, 40, T] log-Mel spectrogram
        returns: [B, embedding_dim] L2-normalized speaker embedding
        """
        x = x.squeeze(1)       # [B, 40, T]
        x = self.stem(x)       # [B, C, T]

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)

        # Concatenate all scales for multi-scale aggregation
        x = torch.cat([f1, f2, f3], dim=1)  # [B, 3C, T]
        x = self.aggregate(x)                # [B, C, T]

        x = self.asp(x)                      # [B, 2C]
        x = self.embedding(x)               # [B, embedding_dim]

        return F.normalize(x, dim=1)        # L2 normalize → unit sphere

    def forward(self, x):
        emb = self.extract_embedding(x)
        if hasattr(self, 'classifier'):
            return self.classifier(emb)
        return emb


class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin) loss.
    Maximizes angular separation between speaker embeddings.
    Critical for getting the FA < 1/hr requirement.
    """
    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.5, scale: float = 64.0):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Cosine similarity between embeddings and class centers
        cosine = F.linear(embeddings, F.normalize(self.weight))

        # Add angular margin to the correct class
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logit = torch.cos(theta + self.margin)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * target_logit) + ((1 - one_hot) * cosine)

        return F.cross_entropy(self.scale * output, labels)