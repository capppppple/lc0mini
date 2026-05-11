from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import torch

from engine.network import load_model
from engine.search import mcts_search


def play_game(
    model_path: str | None = None,
    max_plies: int = 160,
    simulations: int = 64,
    temperature: float = 1.0,
) -> list[dict]:
    board = chess.Board()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device) if model_path else None
    examples: list[dict] = []

    while not board.is_game_over(claim_draw=True) and len(examples) < max_plies:
        result = mcts_search(
            board,
            model=model,
            device=device,
            simulations=simulations,
            temperature=temperature,
            exploration_noise=True,
        )
        examples.append({"fen": board.fen(), "policy": result.policy, "visits": result.visits})
        move = result.move
        board.push(move)

    result = board.result(claim_draw=True)
    value = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)

    for item in examples:
        turn_is_white = " w " in item["fen"]
        item["value"] = value if turn_is_white else -value
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--out", default="data/selfplay.jsonl")
    parser.add_argument("--model", default=None)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as file:
        for _ in range(args.games):
            for example in play_game(
                args.model,
                max_plies=args.max_plies,
                simulations=args.simulations,
                temperature=args.temperature,
            ):
                file.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    main()
