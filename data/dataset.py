"""
dataset.py

언어모델 학습용 Dataset 클래스.
(7_2 torch.utils.data.Dataset 수업에서 다룬 패턴을 그대로 따른다: __init__, __len__, __getitem__)

전처리된 토큰 시퀀스(.bin 파일, uint16 배열)를 memory-map으로 읽어서
(input, target) 쌍 -- target은 input을 한 칸씩 미래로 옮긴 것 -- 을 반환한다.

예) tokens = [10, 25, 7, 99, 3], block_size=3 일 때 하나의 샘플:
    input  = [10, 25, 7]
    target = [25, 7, 99]   (= input을 한 칸 shift)
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class WikiTextDataset(Dataset):
    """전처리된 토큰 바이너리 파일로부터 (input, target) 쌍을 만들어내는 Dataset."""

    def __init__(self, bin_path: str, block_size: int):
        super().__init__()
        assert os.path.exists(bin_path), f"전처리된 파일을 찾을 수 없습니다: {bin_path}"

        # np.memmap: 전체 파일을 메모리에 올리지 않고 필요한 부분만 읽는다.
        # WikiText 전체를 한번에 RAM에 올리면 Colab 환경에서 메모리 부담이 크기 때문에 사용.
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        # 마지막 block_size+1 토큰 구간은 target을 만들 수 없으므로 제외
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int):
        chunk = self.data[idx: idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

    def __repr__(self):
        return (
            f"WikiTextDataset(num_tokens={len(self.data)}, "
            f"block_size={self.block_size}, num_samples={len(self)})"
        )
