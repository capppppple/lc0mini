from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from engine.encoding import board_to_tensor, POLICY_SIZE
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
        policy = torch.tensor(row["move"], dtype=torch.long)
        value = torch.tensor(row["value"], dtype=torch.float32)
        return x, policy, value


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = JsonlChessDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = MiniChessNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, policy_target, value_target in loader:
            x = x.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)

            policy_logits, value_pred = model(x)
            policy_loss = F.cross_entropy(policy_logits, policy_target)
            value_loss = F.mse_loss(value_pred, value_target)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        print(f"epoch={epoch + 1} loss={total_loss / max(len(loader), 1):.4f}")
        save_checkpoint(model, args.out)


def save_checkpoint(model: MiniChessNet, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "policy_size": POLICY_SIZE}, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/selfplay.jsonl")
    parser.add_argument("--out", default="checkpoints/latest.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

