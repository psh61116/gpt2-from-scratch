"""
config.py

GPT-2(scratch 구현) 모델과 학습 과정에서 사용하는 모든 하이퍼파라미터를
한 곳에서 관리하기 위한 모듈.

- GPTConfig   : 모델 구조(레이어 수, 임베딩 차원, head 수 등)
- TrainConfig : 학습 절차(배치 크기, 학습률, 스텝 수 등)

수업에서 다룬 클래스 설계 방식(생성자에서 기본값을 갖는 attribute를 정의하고,
__repr__으로 상태를 확인할 수 있게 하는 패턴)을 그대로 따른다.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """GPT 모델 아키텍처 하이퍼파라미터.

    Colab 무료 T4(16GB VRAM) 기준으로 메모리에 무리 없이 올라가도록
    GPT-2 'small'(124M)보다 더 작은 규모로 기본값을 설정했다.
    필요하면 n_layer, n_embd 등을 키워서 더 큰 모델로 실험할 수 있다.
    """

    vocab_size: int = 50257          # GPT-2 기본 BPE(tiktoken gpt2) 어휘 수
    block_size: int = 256            # 한 번에 보는 최대 context 길이 (시퀀스 길이)
    n_layer: int = 6                 # Transformer Block 개수
    n_head: int = 6                  # Multi-Head Attention의 head 개수
    n_embd: int = 384                # 임베딩(hidden) 차원, n_embd % n_head == 0 이어야 함
    dropout: float = 0.1             # dropout 비율
    bias: bool = True                # Linear, LayerNorm에 bias 사용 여부

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, (
            f"n_embd({self.n_embd})는 n_head({self.n_head})로 나누어져야 합니다."
        )

    def __repr__(self):
        return (
            f"GPTConfig(vocab_size={self.vocab_size}, block_size={self.block_size}, "
            f"n_layer={self.n_layer}, n_head={self.n_head}, n_embd={self.n_embd}, "
            f"dropout={self.dropout}, bias={self.bias})"
        )


@dataclass
class TrainConfig:
    """학습 루프 하이퍼파라미터."""

    # 데이터
    dataset_name: str = "wikitext-2-raw-v1"   # HuggingFace datasets 식별자
    data_dir: str = "data/wikitext"

    # 배치 / 시퀀스
    batch_size: int = 32              # T4 16GB 기준 block_size=256일 때 안전한 값
    grad_accum_steps: int = 4         # gradient accumulation (effective batch size 키우기 용)

    # 옵티마이저
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # 스케줄
    warmup_steps: int = 200
    max_steps: int = 3000
    lr_decay_steps: int = 3000

    # 평가 / 로깅
    eval_interval: int = 200
    eval_iters: int = 50
    log_interval: int = 20

    # 기타
    seed: int = 1337
    device: str = "cuda"              # Colab에서는 자동으로 cuda 사용, 없으면 train.py에서 cpu로 fallback
    dtype: str = "bfloat16"           # T4는 bf16 미지원이므로 train.py에서 fp16/fp32로 자동 전환
    out_dir: str = "checkpoints"
    compile_model: bool = False        # torch.compile 사용 여부 (Colab T4에서는 끄는 것을 권장)

    def __repr__(self):
        return (
            f"TrainConfig(dataset_name={self.dataset_name!r}, batch_size={self.batch_size}, "
            f"learning_rate={self.learning_rate}, max_steps={self.max_steps})"
        )
