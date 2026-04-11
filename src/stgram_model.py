import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

BOTTLENECK_SETTING = [[2, 128, 2, 2], [4, 128, 2, 2], [4, 128, 2, 2]]


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super().__init__()
        self.connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1,
                      groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return x + self.conv(x) if self.connect else self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super().__init__()
        self.linear = linear
        self.conv = nn.Conv2d(inp, oup, k, s, p,
                              groups=inp if dw else 1, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x if self.linear else self.prelu(x)


def _spatial_after_strides(h, w, n=4):
    for _ in range(n):
        h = (h - 1) // 2 + 1
        w = (w - 1) // 2 + 1
    return h, w


class MobileFaceNet(nn.Module):
    def __init__(self, num_class, n_mels=128, num_frames=313,
                 bottleneck_setting=BOTTLENECK_SETTING):
        super().__init__()
        self.conv1 = ConvBlock(2, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)
        self.inplanes = 64
        self.blocks = self._make_layer(Bottleneck, bottleneck_setting)
        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)
        h, w = _spatial_after_strides(n_mels, num_frames)
        self.linear7 = ConvBlock(512, 512, (h, w), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)
        self.fc_out = nn.Linear(128, num_class)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                layers.append(block(self.inplanes, c, s if i == 0 else 1, t))
                self.inplanes = c
        return nn.Sequential(*layers)

    def forward(self, x, label=None):
        x = self.dw_conv1(self.conv1(x))
        x = self.conv2(self.blocks(x))
        x = self.linear1(self.linear7(x))
        feature = x.view(x.size(0), -1)
        return self.fc_out(feature), feature


class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024,
                 hop_len=512, num_frames=313):
        super().__init__()
        self.conv_extractor = nn.Conv1d(
            1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                nn.LayerNorm(num_frames),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for _ in range(num_layer)])

    def forward(self, x):
        return self.conv_encoder(self.conv_extractor(x))


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200,
                 s=30.0, m=0.50, sub=1):
        super().__init__()
        self.s, self.m, self.sub = s, m, sub
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features * sub, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if self.sub > 1:
            cosine = cosine.view(-1, self.out_features, self.sub)
            cosine, _ = torch.max(cosine, dim=2)
        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class STgramMFN(nn.Module):
    def __init__(self, num_classes, c_dim=128, win_len=1024, hop_len=512,
                 n_mels=128, num_frames=313,
                 use_arcface=True, m=0.5, s=30, sub=1):
        super().__init__()
        self.arcface = ArcMarginProduct(
            in_features=128, out_features=num_classes, m=m, s=s, sub=sub
        ) if use_arcface else None
        self.tgramnet = TgramNet(
            mel_bins=c_dim, win_len=win_len,
            hop_len=hop_len, num_frames=num_frames)
        self.mobilefacenet = MobileFaceNet(
            num_class=num_classes, n_mels=n_mels, num_frames=num_frames)

    def forward(self, x_wav, x_mel, label=None):
        x_wav, x_mel = x_wav.unsqueeze(1), x_mel.unsqueeze(1)
        x_t = self.tgramnet(x_wav).unsqueeze(1)
        x = torch.cat((x_mel, x_t), dim=1)
        out, feature = self.mobilefacenet(x, label)
        if self.arcface is not None and label is not None:
            out = self.arcface(feature, label)
        return out, feature
