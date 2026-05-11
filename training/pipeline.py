from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import torch

from engine.network import load_model
from training.evaluate import evaluate
from training.replay import build_replay_buffer, collect_selfplay_files
from training.self_play import play_game
from training.train import train


def write_examples(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example) + "\n")


def summarize_examples(examples: list[dict]) -> dict:
    if not examples:
        return {"positions": 0}

    game_meta = examples[-1].get("game", {})
    values = [float(item.get("value", 0.0)) for item in examples]
    return {
        "positions": len(examples),
        "termination": game_meta.get("termination", "unknown"),
        "result": game_meta.get("result", "unknown"),
        "material_score": game_meta.get("material_score"),
        "adjudicated_value": game_meta.get("adjudicated_value"),
        "avg_abs_value": sum(abs(value) for value in values) / len(values),
    }


def run_iteration(args: argparse.Namespace, iteration: int) -> dict:
    run_dir = Path(args.work_dir) / f"iter_{iteration:04d}"
    data_path = run_dir / "selfplay.jsonl"
    replay_path = run_dir / "replay.jsonl"
    candidate_path = run_dir / "candidate.pt"
    eval_path = run_dir / "eval.json"
    best_path = Path(args.best)
    baseline_path = str(best_path) if best_path.exists() else None

    print(f"iteration={iteration} run_dir={run_dir}")
    print(f"baseline={baseline_path or 'none'}")
    if data_path.exists():
        data_path.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = baseline_path if baseline_path else None
    model = load_model(model_path, device) if model_path else None
    game_summaries = []
    for game_index in range(args.games):
        examples = play_game(
            model=model,
            device=device,
            max_plies=args.max_plies,
            simulations=args.simulations,
            temperature=args.temperature,
            adjudicate_material=not args.no_material_adjudication,
            adjudicate_threshold=args.adjudicate_threshold,
            adjudicate_scale=args.adjudicate_scale,
        )
        write_examples(data_path, examples)
        game_summary = summarize_examples(examples)
        game_summaries.append(game_summary)
        print(
            f"selfplay_game={game_index + 1}/{args.games} "
            f"positions={game_summary['positions']} "
            f"termination={game_summary['termination']} "
            f"material={game_summary.get('material_score')} "
            f"value={game_summary.get('adjudicated_value')}"
        )

    replay_files = collect_selfplay_files(args.work_dir, args.replay_window)
    replay_info = build_replay_buffer(
        files=replay_files,
        out=str(replay_path),
        max_positions=args.max_replay_positions,
        seed=args.seed,
    )
    print(
        f"replay_positions={replay_info['positions']} "
        f"files={len(replay_info['files'])} sampled={replay_info['sampled']}"
    )

    train_args = argparse.Namespace(
        data=str(replay_path),
        out=str(candidate_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=baseline_path if args.resume_from_best and baseline_path else None,
        channels=args.channels,
        blocks=args.blocks,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        clip_grad=args.clip_grad,
        seed=args.seed,
        amp=args.amp,
    )
    train(train_args)

    if baseline_path:
        eval_result = evaluate(
            candidate_path=str(candidate_path),
            baseline_path=baseline_path,
            games=args.eval_games,
            simulations=args.eval_simulations,
            max_plies=args.max_plies,
            seed=args.seed,
        )
    else:
        eval_result = {
            "candidate": str(candidate_path),
            "baseline": None,
            "games": 0,
            "score": 1.0,
            "win_rate": 1.0,
            "simulations": args.eval_simulations,
        }

    eval_path.write_text(json.dumps(eval_result, indent=2), encoding="utf-8")
    promoted = eval_result["win_rate"] >= args.promote_threshold
    if promoted:
        best_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, best_path)
        print(f"promoted=true best={best_path}")
    else:
        print(f"promoted=false threshold={args.promote_threshold}")

    summary = {
        "iteration": iteration,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "data": str(data_path),
        "selfplay": {
            "games": args.games,
            "summaries": game_summaries,
            "terminations": count_terminations(game_summaries),
        },
        "replay": replay_info,
        "candidate": str(candidate_path),
        "best": str(best_path),
        "promoted": promoted,
        "eval": eval_result,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def count_terminations(game_summaries: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for item in game_summaries:
        key = str(item.get("termination", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--work-dir", default="runs")
    parser.add_argument("--best", default="checkpoints/best.pt")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--eval-simulations", type=int, default=32)
    parser.add_argument("--promote-threshold", type=float, default=0.55)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no-material-adjudication", action="store_true")
    parser.add_argument("--adjudicate-threshold", type=float, default=1.0)
    parser.add_argument("--adjudicate-scale", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume-from-best", action="store_true")
    parser.add_argument("--replay-window", type=int, default=5)
    parser.add_argument("--max-replay-positions", type=int, default=0)
    args = parser.parse_args()

    summaries = []
    for iteration in range(1, args.iterations + 1):
        summaries.append(run_iteration(args, iteration))

    summary_path = Path(args.work_dir) / "pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"pipeline_summary={summary_path}")


if __name__ == "__main__":
    main()
