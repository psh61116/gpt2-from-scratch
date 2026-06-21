"""
test_model.py

모델이 의도한 대로 동작하는지 빠르게 확인하는 smoke test.
(unittest 프레임워크 없이, 수업에서 다룬 방식대로 print + assert로 단순하게 검증)

실행 방법:
    python test_model.py

torch가 설치된 환경(로컬 GPU/CPU 또는 Colab)에서 실행하면 됩니다.
"""

import torch

from config import GPTConfig
from model import GPT


def test_forward_shape():
    """forward pass의 출력 shape이 (batch, seq_len, vocab_size)인지 확인."""
    cfg = GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg)

    batch_size, seq_len = 4, 16
    idx = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx)
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size), (
        f"예상 shape (4, 16, 100)과 다름: {logits.shape}"
    )
    assert loss is None, "targets를 주지 않았으므로 loss는 None이어야 합니다."
    print("test_forward_shape: PASS", logits.shape)


def test_loss_computation():
    """targets를 주면 scalar loss가 계산되는지 확인."""
    cfg = GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg)

    idx = torch.randint(0, cfg.vocab_size, (4, 16))
    targets = torch.randint(0, cfg.vocab_size, (4, 16))

    logits, loss = model(idx, targets)
    assert loss is not None
    assert loss.dim() == 0, "loss는 scalar여야 합니다."
    assert loss.item() > 0
    print("test_loss_computation: PASS", loss.item())


def test_generate():
    """generate()가 원하는 길이만큼 토큰을 추가로 생성하는지 확인."""
    cfg = GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg)

    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=10, top_k=10)
    assert out.shape == (1, 14), f"예상 shape (1, 14)과 다름: {out.shape}"
    print("test_generate: PASS", out.shape)


def test_causal_mask_blocks_future():
    """causal mask가 미래 토큰을 실제로 차단하는지 확인.

    같은 prefix에 대해 뒤에 다른 토큰을 붙여도, prefix 위치의 logit은
    바뀌지 않아야 한다 (causal하므로 미래의 영향을 받지 않아야 함).
    """
    cfg = GPTConfig(vocab_size=50, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(cfg)
    model.eval()

    prefix = torch.randint(0, cfg.vocab_size, (1, 4))
    suffix_a = torch.randint(0, cfg.vocab_size, (1, 4))
    suffix_b = torch.randint(0, cfg.vocab_size, (1, 4))

    seq_a = torch.cat([prefix, suffix_a], dim=1)
    seq_b = torch.cat([prefix, suffix_b], dim=1)

    with torch.no_grad():
        logits_a, _ = model(seq_a)
        logits_b, _ = model(seq_b)

    # prefix 구간(0~3번 위치)의 logit은 suffix가 달라도 동일해야 한다.
    assert torch.allclose(logits_a[:, :4, :], logits_b[:, :4, :], atol=1e-5), (
        "causal mask가 제대로 동작하지 않습니다: 미래 토큰이 과거 위치에 영향을 주고 있습니다."
    )
    print("test_causal_mask_blocks_future: PASS")


def test_weight_tying():
    """token_embedding과 lm_head가 실제로 같은 weight 텐서를 공유하는지 확인."""
    cfg = GPTConfig(vocab_size=50, block_size=8, n_layer=1, n_head=1, n_embd=16)
    model = GPT(cfg)
    assert model.token_embedding.weight is model.lm_head.weight, (
        "weight tying이 적용되지 않았습니다."
    )
    print("test_weight_tying: PASS")


if __name__ == "__main__":
    test_forward_shape()
    test_loss_computation()
    test_generate()
    test_causal_mask_blocks_future()
    test_weight_tying()
    print("\n모든 테스트 통과.")
