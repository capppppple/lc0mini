from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import chess
import torch

from engine.network import MiniChessNet
from engine.network import load_model
from engine.search import choose_move


def play_match_game(
    candidate: MiniChessNet,
    baseline: MiniChessNet | None,
    device: torch.device,
    candidate_is_white: bool,
    simulations: int,
    max_plies: int,
) -> float:
    board = chess.Board()

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        use_candidate = board.turn == chess.WHITE if candidate_is_white else board.turn == chess.BLACK
        model = candidate if use_candidate else baseline
        move = choose_move(board, model=model, device=device, simulations=simulations)
        board.push(move)

    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0 if candidate_is_white else 0.0
    if result == "0-1":
        return 0.0 if candidate_is_white else 1.0
    return 0.5


def evaluate(
    candidate_path: str,
    baseline_path: str | None,
    games: int,
    simulations: int,
    max_plies: int,
    seed: int | None,
) -> dict:
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate = load_model(candidate_path, device)
    baseline = load_model(baseline_path, device) if baseline_path else None
    scores = []
    for game_index in range(games):
        candidate_is_white = game_index % 2 == 0
        score = play_match_game(
            candidate,
            baseline,
            device,
            candidate_is_white,
            simulations,
            max_plies,
        )
        scores.append(score)
        print(
            f"eval_game={game_index + 1}/{games} "
            f"candidate_white={candidate_is_white} score={score}"
        )

    total = sum(scores)
    return {
        "candidate": candidate_path,
        "baseline": baseline_path,
        "games": games,
        "score": total,
        "win_rate": total / max(games, 1),
        "simulations": simulations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = evaluate(
        candidate_path=args.candidate,
        baseline_path=args.baseline,
        games=args.games,
        simulations=args.simulations,
        max_plies=args.max_plies,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
