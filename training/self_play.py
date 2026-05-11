from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import random
from pathlib import Path

import chess
import torch

from engine.network import MiniChessNet
from engine.network import load_model
from engine.search import mcts_search


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}


def material_score(board: chess.Board) -> float:
    score = 0.0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score


def adjudicated_value(
    board: chess.Board,
    result: str,
    adjudicate_material: bool,
    adjudicate_threshold: float,
    adjudicate_scale: float,
) -> tuple[float, dict]:
    if result in {"1-0", "0-1", "1/2-1/2"}:
        value = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)
        return value, {"termination": "game_over", "result": result}

    score = material_score(board)
    if not adjudicate_material or abs(score) < adjudicate_threshold:
        return 0.0, {
            "termination": "max_plies_draw",
            "result": result,
            "material_score": score,
        }

    value = math.tanh(score / max(adjudicate_scale, 0.001))
    return value, {
        "termination": "max_plies_material",
        "result": result,
        "material_score": score,
        "adjudicated_value": value,
    }


def play_game(
    model_path: str | None = None,
    model: MiniChessNet | None = None,
    device: torch.device | None = None,
    max_plies: int = 160,
    simulations: int = 64,
    mcts_batch_size: int = 8,
    temperature: float = 1.0,
    temperature_drop_ply: int = 30,
    temperature_final: float = 0.15,
    adjudicate_material: bool = True,
    adjudicate_threshold: float = 1.0,
    adjudicate_scale: float = 8.0,
    store_visits: bool = False,
) -> list[dict]:
    board = chess.Board()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model or (load_model(model_path, device) if model_path else None)
    examples: list[dict] = []

    while not board.is_game_over(claim_draw=True) and len(examples) < max_plies:
        current_temperature = temperature_for_ply(
            board.ply(),
            start_temperature=temperature,
            drop_ply=temperature_drop_ply,
            final_temperature=temperature_final,
        )
        result = mcts_search(
            board,
            model=model,
            device=device,
            simulations=simulations,
            temperature=current_temperature,
            exploration_noise=True,
            mcts_batch_size=mcts_batch_size,
        )
        example = {"fen": board.fen(), "policy": result.policy, "temperature": current_temperature}
        if store_visits:
            example["visits"] = result.visits
        examples.append(example)
        move = result.move
        board.push(move)

    result = board.result(claim_draw=True)
    value, metadata = adjudicated_value(
        board,
        result,
        adjudicate_material=adjudicate_material,
        adjudicate_threshold=adjudicate_threshold,
        adjudicate_scale=adjudicate_scale,
    )

    for item in examples:
        turn_is_white = " w " in item["fen"]
        item["value"] = value if turn_is_white else -value
        item["game"] = metadata
    return examples


def temperature_for_ply(
    ply: int,
    start_temperature: float,
    drop_ply: int,
    final_temperature: float,
) -> float:
    if drop_ply <= 0 or ply >= drop_ply:
        return final_temperature

    progress = ply / drop_ply
    return start_temperature + (final_temperature - start_temperature) * progress


def play_game_worker(config: dict) -> list[dict]:
    seed = config.get("seed")
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    device = torch.device("cpu")
    model = load_model(config["model_path"], device) if config.get("model_path") else None
    return play_game(
        model=model,
        device=device,
        max_plies=config["max_plies"],
        simulations=config["simulations"],
        mcts_batch_size=config["mcts_batch_size"],
        temperature=config["temperature"],
        temperature_drop_ply=config["temperature_drop_ply"],
        temperature_final=config["temperature_final"],
        adjudicate_material=config["adjudicate_material"],
        adjudicate_threshold=config["adjudicate_threshold"],
        adjudicate_scale=config["adjudicate_scale"],
        store_visits=config["store_visits"],
    )


def generate_games(
    games: int,
    model_path: str | None,
    model: MiniChessNet | None,
    device: torch.device,
    workers: int,
    max_plies: int,
    simulations: int,
    mcts_batch_size: int,
    temperature: float,
    temperature_drop_ply: int,
    temperature_final: float,
    adjudicate_material: bool,
    adjudicate_threshold: float,
    adjudicate_scale: float,
    store_visits: bool,
    seed: int | None = None,
):
    if workers <= 1:
        for game_index in range(games):
            yield game_index, play_game(
                model=model,
                device=device,
                max_plies=max_plies,
                simulations=simulations,
                mcts_batch_size=mcts_batch_size,
                temperature=temperature,
                temperature_drop_ply=temperature_drop_ply,
                temperature_final=temperature_final,
                adjudicate_material=adjudicate_material,
                adjudicate_threshold=adjudicate_threshold,
                adjudicate_scale=adjudicate_scale,
                store_visits=store_visits,
            )
        return

    config = {
        "model_path": model_path,
        "max_plies": max_plies,
        "simulations": simulations,
        "mcts_batch_size": mcts_batch_size,
        "temperature": temperature,
        "temperature_drop_ply": temperature_drop_ply,
        "temperature_final": temperature_final,
        "adjudicate_material": adjudicate_material,
        "adjudicate_threshold": adjudicate_threshold,
        "adjudicate_scale": adjudicate_scale,
        "store_visits": store_visits,
    }
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for game_index in range(games):
            worker_config = dict(config)
            worker_config["seed"] = None if seed is None else seed + game_index
            futures[executor.submit(play_game_worker, worker_config)] = game_index
        for future in as_completed(futures):
            yield futures[future], future.result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--out", default="data/selfplay.jsonl")
    parser.add_argument("--model", default=None)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--mcts-batch-size", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-drop-ply", type=int, default=30)
    parser.add_argument("--temperature-final", type=float, default=0.15)
    parser.add_argument("--self-play-workers", type=int, default=1)
    parser.add_argument("--no-material-adjudication", action="store_true")
    parser.add_argument("--adjudicate-threshold", type=float, default=1.0)
    parser.add_argument("--adjudicate-scale", type=float, default=8.0)
    parser.add_argument("--store-visits", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device) if args.model else None
    with out_path.open("a", encoding="utf-8") as file:
        for game_index, examples in generate_games(
            games=args.games,
            model_path=args.model,
            model=model,
            device=device,
            workers=args.self_play_workers,
            max_plies=args.max_plies,
            simulations=args.simulations,
            mcts_batch_size=args.mcts_batch_size,
            temperature=args.temperature,
            temperature_drop_ply=args.temperature_drop_ply,
            temperature_final=args.temperature_final,
            adjudicate_material=not args.no_material_adjudication,
            adjudicate_threshold=args.adjudicate_threshold,
            adjudicate_scale=args.adjudicate_scale,
            store_visits=args.store_visits,
            seed=args.seed,
        ):
            for example in examples:
                file.write(json.dumps(example) + "\n")
            print(f"game={game_index + 1}/{args.games} written={out_path}")


if __name__ == "__main__":
    main()
