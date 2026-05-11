from __future__ import annotations

import math
import random

import chess
import torch

from engine.encoding import board_to_tensor, index_to_move, move_to_index
from engine.network import MiniChessNet


def choose_move(
    board: chess.Board,
    model: MiniChessNet | None = None,
    device: torch.device | str = "cpu",
    temperature: float = 0.0,
) -> chess.Move:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("No legal moves available")

    if model is None:
        return random.choice(legal_moves)

    with torch.no_grad():
        x = board_to_tensor(board).unsqueeze(0).to(device)
        policy_logits, _ = model(x)
        logits = policy_logits[0].detach().cpu()

    scored = []
    for move in legal_moves:
        scored.append((float(logits[move_to_index(move)]), move))

    if temperature <= 0.0:
        return max(scored, key=lambda item: item[0])[1]

    max_logit = max(score for score, _ in scored)
    weights = [math.exp((score - max_logit) / temperature) for score, _ in scored]
    return random.choices([move for _, move in scored], weights=weights, k=1)[0]


def policy_move_from_index(board: chess.Board, index: int) -> chess.Move | None:
    return index_to_move(index, board)

