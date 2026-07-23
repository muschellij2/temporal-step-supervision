import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two convolutional layers used throughout the U-Net style backbone."""

    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(p_drop),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Encoder block that preserves time resolution and downsamples height."""

    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, p_drop=p_drop)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))

    def forward(self, x):
        feat = self.conv(x)
        down = self.pool(feat)
        return feat, down


class UpBlock(nn.Module):
    """Decoder block that upsamples height and fuses the matching skip feature."""

    def __init__(self, in_ch, skip_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=(2, 1),
            stride=(2, 1)
        )
        self.conv = DoubleConv(out_ch + skip_ch, out_ch, p_drop=p_drop)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            x = F.interpolate(
                x,
                size=(skip.shape[2], skip.shape[3]),
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class StepSeqNet(nn.Module):
    """
    U-Net style dense sequence model.

    Input shape is (B, C, H, T), and output shape is (B, T). The temporal axis is
    preserved so each output value corresponds to one input time step.
    """

    def __init__(
        self,
        height_bins,
        sampling_rate,
        window_sec,
        seq_len=180,
        in_channels=3,
        base_ch=32,
        base_drop_2d=0.1,
        head_drop_1d=0.1,
        use_sigmoid=True
    ):
        super().__init__()
        self.seq_len = seq_len
        self.height_bins = height_bins
        self.sampling_rate = sampling_rate
        self.window_sec = window_sec
        self.use_sigmoid = use_sigmoid

        # Encoder path: preserve temporal length while compressing height.
        self.enc1 = DownBlock(in_channels, base_ch, p_drop=base_drop_2d)
        self.enc2 = DownBlock(base_ch, base_ch * 2, p_drop=base_drop_2d)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4, p_drop=base_drop_2d)

        # Bottleneck with a learnable height-position embedding.
        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8, p_drop=base_drop_2d)

        self.height_enc = nn.Parameter(
            torch.randn(1, base_ch * 8, max(1, height_bins // 8), 1)
        )

        # Decoder path: restore height using skip connections.
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4, p_drop=base_drop_2d)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2, p_drop=base_drop_2d)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch, p_drop=base_drop_2d)

        # Temporal refinement increases local context without changing length.
        self.temporal_expand = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(base_ch),
            nn.GELU(),
        )

        # Collapse height to one row before the 1D prediction head.
        self.h_collapse = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(base_ch),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, None))
        )

        # Produce one step-activation probability per time step.
        self.head = nn.Sequential(
            nn.Conv1d(base_ch, base_ch // 2, kernel_size=5, padding=2, bias=True),
            nn.BatchNorm1d(base_ch // 2),
            nn.GELU(),
            nn.Dropout(head_drop_1d),
            nn.Conv1d(base_ch // 2, 1, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x_img):
        # x_img: (B, C, H, T)
        s1, x = self.enc1(x_img)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)

        x = self.bottleneck(x)

        # Resize height embedding to the current encoded height.
        height = x.shape[2]
        enc = F.interpolate(
            self.height_enc,
            size=(height, 1),
            mode="bilinear",
            align_corners=False
        )
        x = x + enc

        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        x = self.temporal_expand(x)
        x = self.h_collapse(x)
        x = x.squeeze(2)

        out = self.head(x)

        if self.use_sigmoid:
            out = torch.sigmoid(out)

        return out.squeeze(1)


class SmallStepSeqNet(nn.Module):
    """Lightweight rasterized dense sequence model for lower-capacity runs."""

    def __init__(
        self,
        height_bins,
        sampling_rate,
        window_sec,
        seq_len=180,
        in_channels=3,
        base_ch=12,
        drop=0.1,
        use_sigmoid=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.height_bins = height_bins
        self.sampling_rate = sampling_rate
        self.window_sec = window_sec
        self.use_sigmoid = use_sigmoid

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.GELU(),
            nn.Dropout2d(drop),
            nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.head = nn.Sequential(
            nn.Conv1d(base_ch * 2, base_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(base_ch, 1, kernel_size=3, padding=1),
        )

    def forward(self, x_img):
        x = self.features(x_img).squeeze(2)
        out = self.head(x).squeeze(1)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out


class CountRegressorNet(nn.Module):
    """
    CNN count-regression baseline.

    It uses the encoder/bottleneck part of the dense model and predicts one
    non-negative step count per window instead of a dense step sequence.
    """

    def __init__(
        self,
        height_bins,
        sampling_rate,
        window_sec,
        seq_len=180,
        in_channels=3,
        base_ch=32,
        base_drop_2d=0.1,
        head_drop=0.1,
    ):
        super().__init__()
        self.enc1 = DownBlock(in_channels, base_ch, p_drop=base_drop_2d)
        self.enc2 = DownBlock(base_ch, base_ch * 2, p_drop=base_drop_2d)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4, p_drop=base_drop_2d)
        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8, p_drop=base_drop_2d)
        self.height_enc = nn.Parameter(
            torch.randn(1, base_ch * 8, max(1, height_bins // 8), 1)
        )
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_ch * 8, base_ch * 4),
            nn.GELU(),
            nn.Dropout(head_drop),
            nn.Linear(base_ch * 4, 1),
            nn.Softplus(),
        )

    def forward(self, x_img):
        _, x = self.enc1(x_img)
        _, x = self.enc2(x)
        _, x = self.enc3(x)
        x = self.bottleneck(x)
        height = x.shape[2]
        enc = F.interpolate(
            self.height_enc,
            size=(height, 1),
            mode="bilinear",
            align_corners=False,
        )
        x = x + enc
        return self.regressor(x).squeeze(1)


class StandardCNNCountRegressorNet(nn.Module):
    """
    Plain CNN count-regression baseline with roughly the same parameter budget as
    StepSeqNet, but without U-Net skips, dense temporal supervision, or learned
    height-position embeddings.
    """

    def __init__(
        self,
        height_bins,
        sampling_rate,
        window_sec,
        seq_len=180,
        in_channels=3,
        head_drop=0.1,
    ):
        super().__init__()
        self.features = nn.Sequential(
            DoubleConv(in_channels, 48, p_drop=0.1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            DoubleConv(48, 96, p_drop=0.1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            DoubleConv(96, 192, p_drop=0.1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            DoubleConv(192, 256, p_drop=0.1),
        )
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(head_drop),
            nn.Linear(512, 1),
            nn.Softplus(),
        )

    def forward(self, x_img):
        x = self.features(x_img)
        return self.regressor(x).squeeze(1)


class RawSeqNet(nn.Module):
    """1D dense sequence model for non-rasterized XYZ windows."""

    def __init__(
        self,
        sampling_rate,
        window_sec,
        seq_len=180,
        in_channels=3,
        base_ch=64,
        head_drop=0.1,
        use_sigmoid=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.sampling_rate = sampling_rate
        self.window_sec = window_sec
        self.use_sigmoid = use_sigmoid

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.GELU(),
            nn.Conv1d(base_ch, base_ch, kernel_size=9, padding=8, dilation=2, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.GELU(),
            nn.Dropout(head_drop),
            nn.Conv1d(base_ch, base_ch, kernel_size=9, padding=16, dilation=4, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.GELU(),
            nn.Conv1d(base_ch, base_ch // 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(base_ch // 2),
            nn.GELU(),
            nn.Conv1d(base_ch // 2, 1, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x_img):
        x = x_img.squeeze(2)
        out = self.net(x).squeeze(1)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out


class SmallRawSeqNet(nn.Module):
    """Lightweight 1D dense sequence model for non-rasterized XYZ windows."""

    def __init__(
        self,
        sampling_rate,
        window_sec,
        seq_len=180,
        in_channels=3,
        base_ch=24,
        head_drop=0.1,
        use_sigmoid=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.sampling_rate = sampling_rate
        self.window_sec = window_sec
        self.use_sigmoid = use_sigmoid

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.GELU(),
            nn.Conv1d(base_ch, base_ch, kernel_size=7, padding=6, dilation=2, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.GELU(),
            nn.Dropout(head_drop),
            nn.Conv1d(base_ch, base_ch // 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(base_ch // 2),
            nn.GELU(),
            nn.Conv1d(base_ch // 2, 1, kernel_size=3, padding=1),
        )

    def forward(self, x_img):
        x = x_img.squeeze(2)
        out = self.net(x).squeeze(1)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out


class RawCountRegressorNet(nn.Module):
    """1D CNN count regressor for non-rasterized XYZ windows."""

    def __init__(
        self,
        sampling_rate,
        window_sec,
        seq_len=180,
        in_channels=3,
        base_ch=64,
        head_drop=0.1,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.GELU(),
            nn.Conv1d(base_ch, base_ch, kernel_size=9, padding=8, dilation=2, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.GELU(),
            nn.Dropout(head_drop),
            nn.Conv1d(base_ch, base_ch, kernel_size=9, padding=16, dilation=4, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.GELU(),
            nn.Conv1d(base_ch, base_ch // 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(base_ch // 2),
            nn.GELU(),
        )
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_ch // 2, base_ch // 2),
            nn.GELU(),
            nn.Dropout(head_drop),
            nn.Linear(base_ch // 2, 1),
            nn.Softplus(),
        )

    def forward(self, x_img):
        x = x_img.squeeze(2)
        x = self.features(x)
        return self.regressor(x).squeeze(1)


class StandardRawCNNCountRegressorNet(nn.Module):
    """
    Plain 1D CNN count-regression baseline for raw XYZ windows.

    This is the non-rasterized counterpart of StandardCNNCountRegressorNet:
    stacked convolution/pooling blocks, global average pooling, and one
    non-negative count output.
    """

    def __init__(
        self,
        sampling_rate,
        window_sec,
        seq_len=180,
        in_channels=3,
        head_drop=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.sampling_rate = sampling_rate
        self.window_sec = window_sec
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(48, 96, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(96, 192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(192, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(head_drop),
            nn.Linear(512, 1),
            nn.Softplus(),
        )

    def forward(self, x_img):
        x = x_img.squeeze(2)
        x = self.features(x)
        return self.regressor(x).squeeze(1)
