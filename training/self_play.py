from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess

from engine.encoding import move_to_index
from engine.network import load_model
from engine.search import choose_move


def play_game(model_path: str | None = None, max_plies: int = 160) -> list[dict]:
    board = chess.Board()
    model = load_model(model_path) if model_path else None
    examples: list[dict] = []

    while not board.is_game_over(claim_draw=True) and len(examples) < max_plies:
        move = choose_move(board, model=model, temperature=1.0)
        examples.append({"fen": board.fen(), "move": move_to_index(move)})
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
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as file:
        for _ in range(args.games):
            for example in play_game(args.model):
                file.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    main()

