#!/usr/bin/env python3
"""Torch model definitions for QUARTZ."""

from __future__ import annotations

import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r),
            nn.ReLU(),
            nn.Linear(ch // r, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = x.mean(dim=(2, 3))
        return x * self.fc(w).unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, ch, se=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )
        self.se = SEBlock(ch) if se else None

    def forward(self, x):
        out = self.net(x)
        if self.se:
            out = self.se(out)
        return x + out


class AlphaZeroNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ch, bs = cfg["filters"], cfg["board"]
        n2 = bs * bs
        self.input_conv = nn.Sequential(
            nn.Conv2d(cfg["ch"], ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
        )
        blocks = [ResBlock(ch, se=(i >= cfg["blocks"] - 2)) for i in range(cfg["blocks"])]
        self.tower = nn.Sequential(*blocks)
        self.p_head = nn.Sequential(
            nn.Conv2d(ch, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.p_fc = nn.Linear(4 * n2, cfg["actions"])
        self.v_head = nn.Sequential(nn.Conv2d(ch, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.v_fc = nn.Sequential(
            nn.Linear(32, cfg["vh"]),
            nn.ReLU(),
            nn.Linear(cfg["vh"], 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.tower(self.input_conv(x))
        p = self.p_head(h).reshape(h.size(0), -1)
        p = self.p_fc(p)
        v = self.v_head(h).mean(dim=(2, 3))
        v = self.v_fc(v).squeeze(-1)
        return p, v
