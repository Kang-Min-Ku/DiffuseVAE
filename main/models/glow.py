import torch
import torch.nn as nn
import torch.nn.functional as F

class DiagGaussian(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(dim))
        self.log_std = nn.Parameter(torch.zeros(dim))

    def forward(self, x=None):
        if x is None:
            return self.mean, torch.exp(self.log_std)
        log_prob = -0.5 * ((x - self.mean) ** 2 * torch.exp(-2 * self.log_std) + 2 * self.log_std + torch.log(torch.tensor(2 * torch.pi)))
        return log_prob.sum(dim=1)

class Split(nn.Module):
    def forward(self, x):
        z1, z2 = x.chunk(2, dim=1)
        return z1, z2

class ConvNet2d(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.0),
            nn.Conv2d(256, 1, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.0),
            nn.Conv2d(1, output_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)

class AffineCoupling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.param_map = ConvNet2d(channels // 2, channels)

    def forward(self, x):
        z1, z2 = x
        params = self.param_map(z1)
        scale, shift = params.chunk(2, dim=1)
        z2 = z2 * torch.exp(scale) + shift
        return torch.cat([z1, z2], dim=1)

class AffineCouplingBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.split = Split()
        self.coupling = AffineCoupling(channels)

    def forward(self, x):
        z1, z2 = self.split(x)
        x = self.coupling((z1, z2))
        return x

class Invertible1x1Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(channels, channels))
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        _, _, h, w = x.size()
        weight = self.weight.unsqueeze(2).unsqueeze(3)
        return F.conv2d(x, weight)

class ActNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return x * self.scale + self.shift

class GlowBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.flows = nn.ModuleList([
            AffineCouplingBlock(channels),
            Invertible1x1Conv(channels),
            ActNorm(channels)
        ])

    def forward(self, x):
        for flow in self.flows:
            x = flow(x)
        return x

class NormalizingFlow(nn.Module):
    def __init__(self, channels, num_blocks):
        super().__init__()
        self.q0 = DiagGaussian(channels)
        self.flows = nn.ModuleList([GlowBlock(channels) for _ in range(num_blocks)])

    def forward(self, x):
        for flow in self.flows:
            x = flow(x)
        return x

class Glow(nn.Module):
    def __init__(self, channels, num_blocks):
        super().__init__()
        self.model = NormalizingFlow(channels, num_blocks)

    def forward(self, x):
        return self.model(x)

    def forward_kld(self, x):
        z = x
        log_det_jacobian = 0.0

        for flow in self.model.flows:
            z = flow(z)
            if hasattr(flow, "log_det_jacobian"):
                log_det_jacobian += flow.log_det_jacobian

        mean, std = self.model.q0()
        prior_log_prob = -0.5 * ((z - mean) ** 2 / (std ** 2) + 2 * torch.log(std) + torch.log(2 * torch.tensor(torch.pi))).sum(dim=[1, 2, 3])

        kld_loss = -(prior_log_prob + log_det_jacobian).mean()
        return kld_loss

    def inverse(self, z):
        x = z
        for flow in reversed(self.model.flows):
            if hasattr(flow, "inverse"):
                x = flow.inverse(x)
        return x, None
