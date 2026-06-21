"""
evaluate.py

학습된 체크포인트를 test split에 대해 평가하여 perplexity를 계산하는 스크립트.

Perplexity = exp(평균 cross-entropy loss)
값이 낮을수록 모델이 다음 토큰을 더 잘 예측한다는 의미.

실행 방법:
    python evaluate.py --ckpt checkpoints/best_model.pt
"""

import os
import math
import argparse

import torch
from torch.utils.data import DataLoader

from model import GPT
from data import WikiTextDataset


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="GPT 모델 perplexity 평가")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--data_dir", type=str, default="data/wikitext")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(args.ckpt, map_location=device)
    gpt_cfg = checkpoint["gpt_config"]
    model = GPT(gpt_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    bin_path = os.path.join(args.data_dir, f"{args.split}.bin")
    dataset = WikiTextDataset(bin_path, gpt_cfg.block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    avg_loss = evaluate(model, loader, device)
    perplexity = math.exp(avg_loss)

    print(f"[{args.split}] average loss = {avg_loss:.4f}")
    print(f"[{args.split}] perplexity  = {perplexity:.2f}")


if __name__ == "__main__":
    main()
