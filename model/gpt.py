"""
gpt.py

GPT-2 아키텍처를 처음부터(from scratch) 구현한 메인 클래스.

전체 forward 흐름:
    1) 토큰 임베딩(token embedding) + 위치 임베딩(positional embedding)
    2) N개의 Transformer Block을 순서대로 통과
    3) 최종 LayerNorm
    4) vocab_size 크기로 projection (lm_head) -> 다음 토큰에 대한 logit

수업에서 다룬 클래스 설계 관례를 따른다:
    - __init__에서 서브모듈들을 등록 (4_2 torch.nn 내부구조 수업 참고)
    - __repr__으로 모델 요약을 사람이 읽기 쉽게 출력 (pfhedge Hedger의 __repr__ 출력 스타일 참고)
    - 파라미터 수 계산 등 유틸리티 메서드 제공
"""

import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.block import Block


class GPT(nn.Module):
    """Decoder-only Transformer 언어모델 (GPT-2 축소판)."""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # 토큰 임베딩 테이블: (vocab_size, n_embd)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # 위치 임베딩 테이블: (block_size, n_embd)  -- GPT-2는 학습되는 절대 위치 임베딩 사용
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer Block을 n_layer개 쌓는다.
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying: 입력 토큰 임베딩과 출력 projection의 가중치를 공유한다.
        # (GPT-2 원 논문 및 대부분의 언어모델 구현에서 사용하는 표준 기법으로,
        #  파라미터 수를 크게 줄이면서 성능 저하는 거의 없다.)
        self.token_embedding.weight = self.lm_head.weight

        # 가중치 초기화
        self.apply(self._init_weights)
        # GPT-2 논문의 residual projection 초기화 스케일링 (1/sqrt(2*n_layer))
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        idx     : (batch, seq_len) 정수 토큰 인덱스
        targets : (batch, seq_len) 다음 토큰 정답 인덱스 (없으면 loss는 None)

        반환값: (logits, loss)
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"입력 시퀀스 길이({T})가 block_size({self.config.block_size})를 초과했습니다."
        )

        positions = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)

        tok_emb = self.token_embedding(idx)          # (B, T, n_embd)
        pos_emb = self.position_embedding(positions)  # (T, n_embd) -> broadcasting으로 (B,T,n_embd)와 더해짐
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross entropy는 (N, vocab_size) vs (N,) 형태를 요구하므로 flatten
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """Autoregressive하게 토큰을 하나씩 생성한다.

        idx: (batch, seq_len) 시작 컨텍스트
        반환: (batch, seq_len + max_new_tokens)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # context가 block_size를 넘으면 뒤쪽 block_size 만큼만 사용
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 마지막 위치의 logit만 사용

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx

    def get_num_params(self, non_embedding: bool = True) -> int:
        """전체 파라미터 수를 반환한다. weight tying 때문에 position embedding만 별도로 뺀다."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding.weight.numel()
        return n_params

    def configure_optimizer(self, weight_decay: float, learning_rate: float, betas: tuple, device_type: str):
        """파라미터를 2D 이상(weight)과 1D(bias, LayerNorm)로 나누어
        weight decay를 선택적으로 적용하는 AdamW optimizer를 생성한다.
        (bias/LayerNorm 가중치에는 weight decay를 주지 않는 것이 표준 관행)
        """
        decay_params = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # PyTorch 버전에 따라 fused AdamW 지원 여부가 다르므로 안전하게 체크
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    def __repr__(self):
        n_params = self.get_num_params() / 1e6
        return (
            f"GPT(\n"
            f"  n_layer={self.config.n_layer}, n_head={self.config.n_head}, "
            f"n_embd={self.config.n_embd}, block_size={self.config.block_size}\n"
            f"  vocab_size={self.config.vocab_size}\n"
            f"  num_params={n_params:.2f}M\n"
            f")"
        )
