"""
attention.py

GPT-2의 핵심인 Causal Multi-Head Self-Attention을 처음부터 구현한다.

수업에서 배운 두 가지를 그대로 활용한다.
    1) `register_buffer` 패턴 (5_3 registry_pattern, micrograd 수업 참고)
       -> causal mask는 학습 파라미터가 아니므로 buffer로 등록한다.
    2) broadcasting (3_1 broadcasting, softmax 수업 참고)
       -> attention score 계산과 softmax에 broadcasting을 적극 활용한다.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Decoder-only Transformer에서 사용하는 causal(미래를 보지 못하는) self-attention.

    입력 x: (batch, seq_len, n_embd)
    출력  : (batch, seq_len, n_embd)  -- 입력과 동일한 shape
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Q, K, V를 한 번의 Linear 연산으로 동시에 계산 (성능상 유리)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 여러 head의 출력을 다시 합친 뒤 거치는 projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 미래 토큰을 보지 못하게 하는 하삼각(lower-triangular) causal mask.
        # 학습되는 파라미터가 아니라 buffer로 등록 -> state_dict에는 포함되지만
        # optimizer.step()의 대상은 아니다. (5_3 registry_pattern과 동일한 사상)
        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "causal_mask",
            causal_mask.view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, seq_len(time), n_embd(channels)

        # (B, T, 3*C) -> 3개의 (B, T, C)로 분리
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention: softmax(QK^T / sqrt(d_k)) V
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 미래 위치는 -inf로 masking 한 뒤 softmax를 취하면 가중치가 0이 된다.
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        out = attn_weights @ v

        # 다시 (B, T, C) 형태로 합치기
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.resid_dropout(self.c_proj(out))
        return out
