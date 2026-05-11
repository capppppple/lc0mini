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
    parser.add_argument("--simulations", type=int, default=64, help="MCTS simulations per move")
    parser.add_argument("--mcts-batch-size", type=int, default=8, help="Neural eval batch size inside MCTS")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device) if args.model else None
    board = chess.Board()
    simulations_default = args.simulations
    mcts_batch_size = args.mcts_batch_size

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        command = line.strip()
        if command == "uci":
            print("id name lc0mini")
            print("id author capppppple")
            print(f"option name Simulations type spin default {simulations_default} min 1 max 10000")
            print(f"option name MCTSBatchSize type spin default {mcts_batch_size} min 1 max 256")
            print("uciok")
        elif command == "isready":
            print("readyok")
        elif command == "ucinewgame":
            board = chess.Board()
        elif command.startswith("position"):
            board = parse_position(command)
        elif command.startswith("setoption"):
            option = parse_setoption(command)
            if option:
                name, value = option
                if name.lower() == "simulations":
                    simulations_default = max(int(value), 1)
                elif name.lower() == "mctsbatchsize":
                    mcts_batch_size = max(int(value), 1)
        elif command.startswith("go"):
            simulations = parse_simulations(command, simulations_default)
            move = choose_move(
                board,
                model=model,
                device=device,
                simulations=simulations,
                mcts_batch_size=mcts_batch_size,
            )
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


def parse_simulations(command: str, default: int) -> int:
    parts = command.split()
    if "nodes" in parts:
        index = parts.index("nodes") + 1
        if index < len(parts):
            return max(int(parts[index]), 1)
    if "movetime" in parts:
        index = parts.index("movetime") + 1
        if index < len(parts):
            return max(min(int(parts[index]) // 20, 1000), 1)
    return default


def parse_setoption(command: str) -> tuple[str, str] | None:
    parts = command.split()
    if "name" not in parts or "value" not in parts:
        return None

    name_start = parts.index("name") + 1
    value_start = parts.index("value")
    if name_start >= value_start or value_start + 1 >= len(parts):
        return None

    name = " ".join(parts[name_start:value_start])
    value = " ".join(parts[value_start + 1 :])
    return name, value


if __name__ == "__main__":
    main()
