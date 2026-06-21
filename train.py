"""
train.py

GPT 모델을 WikiText-2 데이터로 학습시키는 메인 스크립트.

실행 방법:
    python train.py

전제 조건:
    먼저 `python data/prepare_wikitext.py`를 실행해서
    data/wikitext/train.bin, val.bin이 준비되어 있어야 한다.

GPU 사용 방식은 11_2 GPU 수업에서 다룬 패턴을 그대로 사용한다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data, target = data.to(device), target.to(device)
"""

import os
import math
import time
import json

import torch
from torch.utils.data import DataLoader

from config import GPTConfig, TrainConfig
from model import GPT
from data import WikiTextDataset


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Warmup -> Cosine decay -> min_lr 순서의 학습률 스케줄러."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step > cfg.lr_decay_steps:
        return cfg.min_lr
    decay_ratio = (step - cfg.warmup_steps) / (cfg.lr_decay_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, loaders, eval_iters, device):
    """train/val 양쪽에서 평균 loss를 추정한다 (배치 노이즈를 줄이기 위해 여러 번 평균)."""
    out = {}
    model.eval()
    for split, loader in loaders.items():
        losses = torch.zeros(eval_iters)
        loader_iter = iter(loader)
        for i in range(eval_iters):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    gpt_cfg = GPTConfig()
    train_cfg = TrainConfig()

    torch.manual_seed(train_cfg.seed)

    # --- GPU 자동 감지 (11_2 수업과 동일한 패턴) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # T4는 bfloat16을 지원하지 않으므로 자동으로 fp16/fp32 결정
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else torch.float32
    try:
        # PyTorch >= 2.3
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    except TypeError:
        # 구버전 PyTorch fallback
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- 데이터 준비 ---
    data_dir = os.path.join(os.path.dirname(__file__), train_cfg.data_dir.replace("data/", ""))
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        gpt_cfg.vocab_size = meta["vocab_size"]
        print(f"meta.json에서 vocab_size={gpt_cfg.vocab_size} 로드")

    train_dataset = WikiTextDataset(os.path.join(data_dir, "train.bin"), gpt_cfg.block_size)
    val_dataset = WikiTextDataset(os.path.join(data_dir, "val.bin"), gpt_cfg.block_size)
    print(train_dataset)
    print(val_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    # --- 모델 ---
    model = GPT(gpt_cfg).to(device)
    print(model)

    optimizer = model.configure_optimizer(
        weight_decay=train_cfg.weight_decay,
        learning_rate=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        device_type=device.type,
    )

    os.makedirs(train_cfg.out_dir, exist_ok=True)
    best_val_loss = float("inf")

    train_iter = iter(train_loader)
    t0 = time.time()
    history = []

    for step in range(train_cfg.max_steps):
        lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for micro_step in range(train_cfg.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / train_cfg.grad_accum_steps

            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % train_cfg.log_interval == 0:
            dt = time.time() - t0
            print(f"step {step:5d} | loss {accumulated_loss:.4f} | lr {lr:.2e} | {dt:.1f}s")
            t0 = time.time()

        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps - 1:
            losses = estimate_loss(
                model,
                {"train": train_loader, "val": val_loader},
                train_cfg.eval_iters,
                device,
            )
            val_ppl = math.exp(losses["val"])
            print(f"  [eval] step {step}: train_loss={losses['train']:.4f}, "
                  f"val_loss={losses['val']:.4f}, val_perplexity={val_ppl:.2f}")
            history.append({"step": step, **losses, "val_perplexity": val_ppl})

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "gpt_config": gpt_cfg,
                    "step": step,
                    "best_val_loss": best_val_loss,
                }
                ckpt_path = os.path.join(train_cfg.out_dir, "best_model.pt")
                torch.save(checkpoint, ckpt_path)
                print(f"  -> 새로운 best checkpoint 저장: {ckpt_path} (val_loss={best_val_loss:.4f})")

    # 학습 기록을 results/training_log.json 으로 저장 (README/보고서 작성용)
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "training_log.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("학습 완료.")


if __name__ == "__main__":
    main()
