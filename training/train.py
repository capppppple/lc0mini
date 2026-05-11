from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from engine.encoding import POLICY_SIZE, board_to_tensor
from engine.network import MiniChessNet


class JsonlChessDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.rows = []
        with open(path, encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        board = chess.Board(row["fen"])
        x = board_to_tensor(board)
        policy = torch.zeros(POLICY_SIZE, dtype=torch.float32)
        if "policy" in row:
            for move_index, prob in row["policy"].items():
                policy[int(move_index)] = float(prob)
        else:
            policy[int(row["move"])] = 1.0
        policy_sum = float(policy.sum().item())
        if policy_sum > 0:
            policy /= policy_sum
        value = torch.tensor(row["value"], dtype=torch.float32)
        return x, policy, value


def train(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = JsonlChessDataset(args.data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = MiniChessNet(channels=args.channels, blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        arch = checkpoint.get("arch", {})
        if arch:
            model = MiniChessNet(
                channels=int(arch.get("channels", args.channels)),
                blocks=int(arch.get("blocks", args.blocks)),
            ).to(device)
        model.load_state_dict(checkpoint["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))

    print(
        f"device={device} positions={len(dataset)} "
        f"channels={model.channels} blocks={model.blocks} amp={use_amp}"
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        for x, policy_target, value_target in loader:
            x = x.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)

            optimizer.zero_grad()
            amp_context = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
            with amp_context:
                policy_logits, value_pred = model(x)
                log_probs = F.log_softmax(policy_logits, dim=1)
                policy_loss = -(policy_target * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(value_pred, value_target)
                loss = policy_loss + value_loss

            if use_amp:
                scaler.scale(loss).backward()
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

            total_loss += float(loss.item())
            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())

        steps = max(len(loader), 1)
        current_epoch = start_epoch + epoch + 1
        print(
            f"epoch={current_epoch} "
            f"loss={total_loss / steps:.4f} "
            f"policy={total_policy_loss / steps:.4f} "
            f"value={total_value_loss / steps:.4f}"
        )
        save_checkpoint(model, optimizer, args.out, current_epoch, args)


def save_checkpoint(
    model: MiniChessNet,
    optimizer: torch.optim.Optimizer,
    path: str,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "policy_size": POLICY_SIZE,
            "arch": {"channels": model.channels, "blocks": model.blocks},
            "train_args": vars(args),
        },
        out_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/selfplay.jsonl")
    parser.add_argument("--out", default="checkpoints/latest.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
