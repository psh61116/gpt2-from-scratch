"""
mlp.py

Transformer Block 내부의 Position-wise FeedForward Network.
구조: Linear -> GELU -> Linear -> Dropout
(2_1, 8_2 MLP 수업에서 다룬 단순 MLP 구조의 연장선)
"""

import torch.nn as nn


class MLP(nn.Module):
    """GPT-2 스타일 FeedForward 서브레이어.

    hidden 차원을 4배로 확장한 뒤 다시 원래 차원으로 projection 한다.
    활성함수는 GPT-2 원 논문과 동일하게 GELU를 사용한다.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
