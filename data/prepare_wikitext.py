"""
prepare_wikitext.py

HuggingFace `datasets` 라이브러리로 WikiText-2(raw) 데이터셋을 다운로드하고,
GPT-2와 동일한 BPE 토크나이저(tiktoken)로 인코딩하여
학습/검증용 바이너리 파일(train.bin, val.bin)로 저장한다.

tiny Shakespeare(글자 단위, 단일 파일) 대비 WikiText는:
    - 실제 위키백과 문서로 구성된, 훨씬 더 다양한 주제/문체를 포함
    - 공식 train/validation/test 분할 제공
    - GPT-2 원 논문이 사용한 BPE 토크나이저를 그대로 적용 가능 (vocab_size=50257)

실행 방법 (Colab 또는 로컬):
    python data/prepare_wikitext.py

출력:
    data/wikitext/train.bin
    data/wikitext/val.bin
    data/wikitext/meta.json   (토큰 개수, vocab_size 등 메타정보)
"""

import os
import json

import numpy as np


def main():
    # 의존 라이브러리는 함수 내부에서 import -> 단순히 모듈을 불러오기만 할 때는
    # datasets/tiktoken이 없어도 ImportError가 나지 않도록 한다.
    from datasets import load_dataset
    import tiktoken

    out_dir = os.path.join(os.path.dirname(__file__), "wikitext")
    os.makedirs(out_dir, exist_ok=True)

    print("[1/3] WikiText-2(raw) 데이터셋 다운로드 중...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    enc = tiktoken.get_encoding("gpt2")  # GPT-2 원본 BPE 토크나이저, vocab_size=50257

    def encode_split(split_name: str) -> np.ndarray:
        """split('train'/'validation'/'test')의 모든 문서를 이어붙여 토큰화한다."""
        text = "\n".join(
            line for line in dataset[split_name]["text"] if line.strip() != ""
        )
        ids = enc.encode_ordinary(text)  # special token 없이 순수 BPE 인코딩
        ids.append(enc.eot_token)  # 문서(split) 끝을 알리는 end-of-text 토큰
        return np.array(ids, dtype=np.uint16)

    print("[2/3] BPE(tiktoken gpt2) 토큰화 진행 중...")
    train_ids = encode_split("train")
    val_ids = encode_split("validation")

    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    meta = {
        "dataset": "wikitext-2-raw-v1",
        "tokenizer": "tiktoken/gpt2",
        "vocab_size": enc.n_vocab,
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[3/3] 완료.")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
