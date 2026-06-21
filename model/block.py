"""
block.py

하나의 Transformer Decoder Block.

구조 (GPT-2 방식의 Pre-LayerNorm):
    x = x + Attention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))

residual connection을 사용하기 때문에 LayerNorm을 attention/MLP *이전*에
적용하는 것이 학습 안정성 측면에서 유리하다 (GPT-2 논문에서 채택한 방식).
"""

import torch.nn as nn

from model.attention import CausalSelfAttention
from model.mlp import MLP


class Block(nn.Module):
    """단일 Transformer decoder block."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
