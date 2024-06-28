import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data

import torchvision
import torchvision.transforms as transforms

import pandas as pd
import time
from tqdm import tqdm
import random

import plotly.express as px
import plotly.graph_objects as go

from typing import Tuple

from create_data import *
import create_data

device = "cuda" if torch.cuda.is_available() else "cpu"


class CouplingLayer(nn.Module):
    def __init__(self, dim: int = 2):
        super(CouplingLayer, self).__init__()
        self.dim = dim
        self.nn = nn.Sequential(
            nn.Linear(dim // 2, dim), nn.Tanh(), nn.Linear(dim, dim), nn.Tanh()
        )

    def f(self, zl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.nn(zl)
        s, b = out.chunk(2, dim=1)
        return torch.log(s), b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zl, zr = x.chunk(2, dim=1)
        log_s, b = self.f(zl)
        yr = zr * torch.exp(log_s) + b  # affine transformation
        return torch.cat([zl, yr], dim=1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        zl, zr = y.chunk(2, dim=1)
        log_s, b = self.f(zl)
        zr = (zr - b) * torch.exp(-log_s)
        return torch.cat((zl, zr), dim=1)

    def log_det(self, x: torch.Tensor) -> torch.Tensor:
        zl, zr = x.chunk(2, dim=1)
        log_s, b = self.f(zl)
        inverse_s = log_s * -1
        return inverse_s.sum(dim=1)

    def det(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_det(x))


m1 = CouplingLayer()
m2 = CouplingLayer()
m3 = CouplingLayer()

x = torch.randn(10, 2)
y1 = m1(x)
y2 = m2(y1)
y3 = m3(y2)
