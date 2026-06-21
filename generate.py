"""
generate.py

학습된 체크포인트를 불러와 텍스트를 생성(샘플링)하는 스크립트.

실행 방법:
    python generate.py --prompt "The history of artificial intelligence" --max_new_tokens 200
"""

import argparse

import torch
import tiktoken

from model import GPT


def load_model(ckpt_path: str, device: torch.device) -> GPT:
    checkpoint = torch.load(ckpt_path, map_location=device)
    gpt_cfg = checkpoint["gpt_config"]
    model = GPT(gpt_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"체크포인트 로드 완료 (step={checkpoint['step']}, "
          f"val_loss={checkpoint['best_val_loss']:.4f})")
    print(model)
    return model


def main():
    parser = argparse.ArgumentParser(description="학습된 GPT 모델로 텍스트 생성")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--prompt", type=str, default="The history of")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.ckpt, device)
    enc = tiktoken.get_encoding("gpt2")

    start_ids = enc.encode_ordinary(args.prompt)
    idx = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    out_ids = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    generated_text = enc.decode(out_ids[0].tolist())

    print("\n" + "=" * 60)
    print(generated_text)
    print("=" * 60)


if __name__ == "__main__":
    main()
