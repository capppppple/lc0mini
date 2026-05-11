from __future__ import annotations

import argparse
import sys

import chess
import torch

from engine.network import load_model
from engine.search import choose_move


def main() -> None:
    parser = argparse.ArgumentParser(description="lc0mini UCI engine")
    parser.add_argument("--model", default=None, help="Path to a .pt checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device) if args.model else None
    board = chess.Board()

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        command = line.strip()
        if command == "uci":
            print("id name lc0mini")
            print("id author capppppple")
            print("uciok")
        elif command == "isready":
            print("readyok")
        elif command == "ucinewgame":
            board = chess.Board()
        elif command.startswith("position"):
            board = parse_position(command)
        elif command.startswith("go"):
            move = choose_move(board, model=model, device=device)
            print(f"bestmove {move.uci()}")
        elif command == "quit":
            break
        sys.stdout.flush()


def parse_position(command: str) -> chess.Board:
    parts = command.split()
    if len(parts) < 2:
        return chess.Board()

    if parts[1] == "startpos":
        board = chess.Board()
        move_start = 3 if len(parts) > 2 and parts[2] == "moves" else len(parts)
    elif parts[1] == "fen":
        moves_index = parts.index("moves") if "moves" in parts else len(parts)
        board = chess.Board(" ".join(parts[2:moves_index]))
        move_start = moves_index + 1
    else:
        return chess.Board()

    for move_text in parts[move_start:]:
        board.push_uci(move_text)
    return board


if __name__ == "__main__":
    main()

