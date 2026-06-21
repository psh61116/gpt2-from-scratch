# GPT-2 from Scratch — WikiText-2로 학습하기

PyTorch만 사용해서 GPT-2 아키텍처를 **처음부터(from scratch)** 구현하고,
[tiny Shakespeare]가 아닌 **WikiText-2(raw)** 데이터셋으로 직접 학습시킨 프로젝트입니다.

Colab 무료 GPU(T4)에서 바로 돌릴 수 있도록 모델 크기와 학습 설정을 맞췄습니다.

---

## 1. 프로젝트 구조

```
gpt2-from-scratch/
├── README.md                  # 본 파일
├── requirements.txt            # 의존 패키지 목록
├── .gitignore
├── LICENSE
│
├── config/
│   ├── __init__.py
│   └── config.py               # GPTConfig(모델 구조), TrainConfig(학습 설정)
│
├── model/
│   ├── __init__.py
│   ├── attention.py            # Causal Multi-Head Self-Attention
│   ├── mlp.py                  # Position-wise FeedForward(MLP)
│   ├── block.py                # Transformer Decoder Block (Attn + MLP + LayerNorm)
│   └── gpt.py                  # 전체 GPT 모델 (임베딩 + N개 Block + lm_head)
│
├── data/
│   ├── __init__.py
│   ├── prepare_wikitext.py     # WikiText-2 다운로드 + BPE 토큰화 전처리
│   └── dataset.py              # torch.utils.data.Dataset 구현
│
├── train.py                    # 학습 메인 스크립트
├── generate.py                  # 학습된 모델로 텍스트 생성
├── evaluate.py                  # perplexity 평가
│
├── notebooks/
│   └── train_colab.ipynb       # Colab에서 바로 실행 가능한 노트북
│
└── results/
    └── training_log.json        # 학습 후 생성되는 step별 loss/perplexity 기록
```

### 모듈 간 의존 관계

```
config.GPTConfig ──┐
                    ├──> model.gpt.GPT
config.TrainConfig ─┘         │
                               ├──> model.block.Block (×n_layer)
                               │         ├──> model.attention.CausalSelfAttention
                               │         └──> model.mlp.MLP
                               │
data.prepare_wikitext.py ──> data/wikitext/{train,val}.bin
                               │
data.dataset.WikiTextDataset <┘
        │
        ▼
   train.py  ──(학습)──> checkpoints/best_model.pt
        │
        ├──> generate.py (텍스트 생성)
        └──> evaluate.py (perplexity 평가)
```

---

## 2. 모델 아키텍처

GPT-2 원 논문(Decoder-only Transformer)을 그대로 따르되, Colab 무료 T4(16GB VRAM)에서
무리 없이 학습되도록 GPT-2 'small'(124M)보다 작은 규모를 기본값으로 사용합니다.

| 항목 | 값 |
|---|---|
| 레이어 수 (`n_layer`) | 6 |
| Attention head 수 (`n_head`) | 6 |
| 임베딩 차원 (`n_embd`) | 384 |
| context 길이 (`block_size`) | 256 |
| 어휘 크기 (`vocab_size`) | 50,257 (GPT-2 BPE, tiktoken) |
| 파라미터 수 | 약 11M (embedding 제외 기준) |

구조적 특징:
- **Pre-LayerNorm** 방식 (`x = x + Attn(LN(x))`, `x = x + MLP(LN(x))`) — 학습 안정성을 위해 GPT-2가 채택한 방식
- **Weight Tying**: 입력 토큰 임베딩과 출력 `lm_head`의 가중치를 공유하여 파라미터 수 절감
- **Causal mask**는 학습 파라미터가 아닌 `register_buffer`로 등록
- Optimizer는 bias/LayerNorm 가중치에는 weight decay를 적용하지 않는 AdamW 사용

---

## 3. 데이터셋: WikiText-2 (tiny Shakespeare 대신 사용)

과제 요구사항에 따라 tiny Shakespeare가 아닌 **WikiText-2-raw**를 사용했습니다.

| 항목 | tiny Shakespeare | WikiText-2 (본 프로젝트) |
|---|---|---|
| 도메인 | 희곡 대사체 (등장인물 반복) | 위키백과 백과사전 서술 |
| 토큰화 방식 | 글자 단위(char-level)가 흔함 | BPE subword (GPT-2 원본 토크나이저) |
| 어휘 다양성 | 낮음 | 높음 (인물/지명/숫자/전문용어 다양) |
| 분할 | 보통 직접 train/val 분리 | 공식 train/validation/test 제공 |

`data/prepare_wikitext.py`는 HuggingFace `datasets`로 WikiText-2-raw를 내려받고,
GPT-2 원본 BPE 토크나이저(`tiktoken`, vocab_size=50,257)로 인코딩하여
`data/wikitext/train.bin`, `val.bin`으로 저장합니다.

---

## 4. 사용법

### 4-1. 로컬 / 일반 GPU 서버에서

```bash
git clone <이 저장소 URL>
cd gpt2-from-scratch
pip install -r requirements.txt

# 1) 데이터 준비 (WikiText-2 다운로드 + BPE 토큰화)
python data/prepare_wikitext.py

# 2) 학습
python train.py

# 3) 텍스트 생성
python generate.py --prompt "The history of the United States" --max_new_tokens 200

# 4) perplexity 평가
python evaluate.py --ckpt checkpoints/best_model.pt --split val
```

### 4-2. Colab(T4 무료 GPU)에서

`notebooks/train_colab.ipynb`를 Colab에서 열고, 런타임 유형을 **GPU(T4)**로 설정한 뒤
셀을 위에서부터 순서대로 실행하면 됩니다. 노트북 안에서 저장소 클론 → 데이터 준비 →
학습 → 결과 시각화 → 텍스트 생성까지 전체 파이프라인이 자동으로 진행됩니다.

### 4-3. 하이퍼파라미터 수정

`config/config.py`의 `GPTConfig`(모델 구조), `TrainConfig`(학습 절차) 값을 직접 수정하면 됩니다.

```python
from config import GPTConfig, TrainConfig

gpt_cfg = GPTConfig(n_layer=8, n_embd=512)   # 모델을 더 키우고 싶을 때
train_cfg = TrainConfig(max_steps=1000)       # 빠른 데모용으로 줄이고 싶을 때
```

---

## 5. 학습 결과

> **아래 표/그래프는 `train.py` 실행 후 `results/training_log.json`을 기준으로 채워야 하는 영역입니다.**
> 본 저장소에는 실행 가이드와 측정 코드까지 모두 포함되어 있으며,
> 실제 Colab T4에서 `notebooks/train_colab.ipynb`를 실행하면 아래와 동일한 형식의 결과가 생성됩니다.

### 5-1. 학습 곡선

`train.py`가 `eval_interval`마다 train/validation loss와 validation perplexity를 측정하여
`results/training_log.json`에 저장하고, Colab 노트북의 5번 셀에서 아래와 같은 그래프로
시각화합니다 (`results/training_curves.png`).

- 좌측 그래프: step별 train loss / validation loss
- 우측 그래프: step별 validation perplexity

일반적으로 기대되는 경향:
- 초반(warmup 구간, `warmup_steps=200`)에는 loss가 빠르게 감소
- 이후 cosine decay에 따라 학습률이 점진적으로 낮아지며 loss가 완만하게 수렴
- WikiText는 tiny Shakespeare보다 어휘/문장 구조가 다양하므로, 동일한 모델 크기·스텝 수 기준으로
  validation perplexity가 더 높게 나오는 것이 정상입니다 (데이터 자체의 엔트로피가 높기 때문)

### 5-2. 최종 평가 지표

```bash
python evaluate.py --ckpt checkpoints/best_model.pt --split val
```

위 명령 실행 시 다음 형식으로 출력됩니다.

```
[val] average loss = <측정값>
[val] perplexity   = <측정값>
```

### 5-3. 텍스트 생성 예시

```bash
python generate.py --prompt "The history of the United States" --max_new_tokens 200
```

학습이 충분히 진행되면, WikiText 특유의 **백과사전식 서술 문체**(인물 소개, 연도, 지명,
사건 서술 등)가 생성되는 것을 확인할 수 있습니다. tiny Shakespeare로 학습했을 때
나오는 희곡 대사체("KING:", "Exeunt" 등)와는 분명히 다른 스타일이 나오는 것이 본 실험의
핵심 비교 포인트입니다.

---

## 6. 수업 내용과의 연결

이 구현은 학기 중 다룬 아래 개념들을 그대로 응용한 것입니다.

- `class`, dunder method, `__repr__`, `__init__` 설계 패턴
- 밑바닥부터 구현한 MLP / backward (micrograd 실습)
- `nn.Module` 내부 구조 (`torch.nn`이 실제로 어떻게 동작하는지)
- broadcasting과 softmax
- `register_buffer`를 활용한 비학습 파라미터 등록 (registry pattern 실습)
- `torch.utils.data.Dataset`을 활용한 커스텀 데이터셋 구성
- GPU 자동 감지 및 `device.to()` 패턴

---

## 7. 라이선스

[MIT License](LICENSE)
