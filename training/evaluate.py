from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import chess
import torch

from engine.network import MiniChessNet
from engine.network import load_model
from engine.search import choose_move
from training.self_play import material_score


def play_match_game(
    candidate: MiniChessNet,
    baseline: MiniChessNet | None,
    device: torch.device,
    candidate_is_white: bool,
    simulations: int,
    mcts_batch_size: int,
    max_plies: int,
    material_tiebreak_weight: float,
    material_tiebreak_scale: float,
) -> dict:
    board = chess.Board()

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        use_candidate = board.turn == chess.WHITE if candidate_is_white else board.turn == chess.BLACK
        model = candidate if use_candidate else baseline
        move = choose_move(
            board,
            model=model,
            device=device,
            simulations=simulations,
            mcts_batch_size=mcts_batch_size,
        )
        board.push(move)

    result = board.result(claim_draw=True)
    material = material_score(board)
    candidate_material = material if candidate_is_white else -material
    if result == "1-0":
        score = 1.0 if candidate_is_white else 0.0
    elif result == "0-1":
        score = 0.0 if candidate_is_white else 1.0
    else:
        score = 0.5

    is_draw = score == 0.5
    material_bonus = 0.0
    if is_draw and material_tiebreak_weight > 0:
        material_bonus = material_tiebreak_weight * math.tanh(
            candidate_material / max(material_tiebreak_scale, 0.001)
        )
    promotion_score = min(max(score + material_bonus, 0.0), 1.0)

    return {
        "score": score,
        "promotion_score": promotion_score,
        "material_bonus": material_bonus,
        "material_score": material,
        "candidate_material_score": candidate_material,
        "material_tiebreak_applied": is_draw and material_tiebreak_weight > 0,
        "result": result,
        "candidate_color": "white" if candidate_is_white else "black",
        "plies": board.ply(),
    }


def evaluate(
    candidate_path: str,
    baseline_path: str | None,
    games: int,
    simulations: int,
    mcts_batch_size: int,
    max_plies: int,
    seed: int | None,
    material_tiebreak_weight: float = 0.0,
    material_tiebreak_scale: float = 8.0,
) -> dict:
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate = load_model(candidate_path, device)
    baseline = load_model(baseline_path, device) if baseline_path else None
    game_results = []
    for game_index in range(games):
        candidate_is_white = game_index % 2 == 0
        result = play_match_game(
            candidate,
            baseline,
            device,
            candidate_is_white,
            simulations,
            mcts_batch_size,
            max_plies,
            material_tiebreak_weight,
            material_tiebreak_scale,
        )
        game_results.append(result)
        print(
            f"eval_game={game_index + 1}/{games} "
            f"candidate_color={result['candidate_color']} "
            f"score={result['score']} "
            f"promotion_score={result['promotion_score']:.3f} "
            f"material_bonus={result['material_bonus']:.3f} "
            f"result={result['result']} plies={result['plies']}"
        )

    total = sum(item["score"] for item in game_results)
    promotion_total = sum(item["promotion_score"] for item in game_results)
    wins = sum(1 for item in game_results if item["score"] == 1.0)
    draws = sum(1 for item in game_results if item["score"] == 0.5)
    losses = sum(1 for item in game_results if item["score"] == 0.0)
    white_scores = [item["score"] for item in game_results if item["candidate_color"] == "white"]
    black_scores = [item["score"] for item in game_results if item["candidate_color"] == "black"]
    win_rate = total / max(games, 1)
    promotion_rate = promotion_total / max(games, 1)
    return {
        "candidate": candidate_path,
        "baseline": baseline_path,
        "games": games,
        "score": total,
        "win_rate": win_rate,
        "elo_diff": elo_from_score(win_rate),
        "promotion_score": promotion_total,
        "promotion_rate": promotion_rate,
        "promotion_elo_diff": elo_from_score(promotion_rate),
        "material_tiebreak_weight": material_tiebreak_weight,
        "material_tiebreak_scale": material_tiebreak_scale,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "white_score": sum(white_scores) / max(len(white_scores), 1),
        "black_score": sum(black_scores) / max(len(black_scores), 1),
        "results": game_results,
        "simulations": simulations,
        "mcts_batch_size": mcts_batch_size,
    }


def elo_from_score(score: float) -> float:
    clipped = min(max(score, 0.01), 0.99)
    return -400.0 * math.log10(1.0 / clipped - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--mcts-batch-size", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--material-tiebreak-weight", type=float, default=0.0)
    parser.add_argument("--material-tiebreak-scale", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = evaluate(
        candidate_path=args.candidate,
        baseline_path=args.baseline,
        games=args.games,
        simulations=args.simulations,
        mcts_batch_size=args.mcts_batch_size,
        max_plies=args.max_plies,
        seed=args.seed,
        material_tiebreak_weight=args.material_tiebreak_weight,
        material_tiebreak_scale=args.material_tiebreak_scale,
    )
    print(json.dumps(result, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
