from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from engine.network import load_model
from training.evaluate import evaluate
from training.replay import build_replay_buffer, collect_selfplay_files
from training.self_play import generate_games
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
    workers = effective_self_play_workers(args.self_play_workers, model_path, device)
    print(f"self_play_workers={workers}")
    for game_index, examples in generate_games(
        games=args.games,
        model_path=model_path,
        model=model,
        device=device,
        workers=workers,
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
        seed=None if args.seed is None else args.seed + iteration * 100000,
    ):
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

    should_evaluate = baseline_path and (iteration % args.eval_interval == 0)
    if should_evaluate:
        eval_result = evaluate(
            candidate_path=str(candidate_path),
            baseline_path=baseline_path,
            games=args.eval_games,
            simulations=args.eval_simulations,
            mcts_batch_size=args.eval_mcts_batch_size,
            max_plies=args.max_plies,
            seed=args.seed,
            material_tiebreak_weight=args.promotion_material_weight,
            material_tiebreak_scale=args.promotion_material_scale,
        )
    elif baseline_path:
        eval_result = {
            "candidate": str(candidate_path),
            "baseline": baseline_path,
            "games": 0,
            "score": 0.0,
            "win_rate": 0.0,
            "simulations": args.eval_simulations,
            "mcts_batch_size": args.eval_mcts_batch_size,
            "skipped": True,
            "reason": f"eval_interval={args.eval_interval}",
        }
    else:
        eval_result = {
            "candidate": str(candidate_path),
            "baseline": None,
            "games": 0,
            "score": 1.0,
            "win_rate": 1.0,
            "simulations": args.eval_simulations,
            "mcts_batch_size": args.eval_mcts_batch_size,
        }

    eval_path.write_text(json.dumps(eval_result, indent=2), encoding="utf-8")
    promotion_rate = eval_result.get("promotion_rate", eval_result["win_rate"])
    promoted = (not eval_result.get("skipped", False)) and promotion_rate >= args.promote_threshold
    if promoted:
        best_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, best_path)
        print(f"promoted=true best={best_path} promotion_rate={promotion_rate:.3f}")
    else:
        print(f"promoted=false threshold={args.promote_threshold} promotion_rate={promotion_rate:.3f}")

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


def effective_self_play_workers(
    requested_workers: int,
    model_path: str | None,
    device: torch.device,
) -> int:
    if requested_workers <= 1:
        return 1
    if model_path and device.type == "cuda":
        print("parallel_cpu_selfplay_disabled_for_cuda_model=true")
        return 1
    return requested_workers


def count_terminations(game_summaries: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for item in game_summaries:
        key = str(item.get("termination", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=["debug", "fast", "balanced", "strong"],
        default=None,
        help="Apply a speed/quality preset. Explicit CLI args still win.",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--work-dir", default="runs")
    parser.add_argument("--start-iteration", type=int, default=None)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--best", default="checkpoints/best.pt")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--mcts-batch-size", type=int, default=8)
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--eval-simulations", type=int, default=32)
    parser.add_argument("--eval-mcts-batch-size", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--promote-threshold", type=float, default=0.55)
    parser.add_argument("--promotion-material-weight", type=float, default=0.02)
    parser.add_argument("--promotion-material-scale", type=float, default=8.0)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-drop-ply", type=int, default=30)
    parser.add_argument("--temperature-final", type=float, default=0.15)
    parser.add_argument("--self-play-workers", type=int, default=1)
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
    parser.add_argument("--store-visits", action="store_true")
    args = parser.parse_args()
    apply_preset(args, sys.argv[1:])

    start_iteration = resolve_start_iteration(args.work_dir, args.start_iteration, args.restart)
    end_iteration = start_iteration + args.iterations - 1
    print(f"iteration_range={start_iteration}..{end_iteration}")

    summaries = load_pipeline_summary(args.work_dir)
    for iteration in range(start_iteration, end_iteration + 1):
        summaries.append(run_iteration(args, iteration))

    summary_path = Path(args.work_dir) / "pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"pipeline_summary={summary_path}")


def resolve_start_iteration(
    work_dir: str,
    explicit_start: int | None,
    restart: bool,
) -> int:
    if explicit_start is not None:
        return max(explicit_start, 1)
    if restart:
        return 1
    return find_next_iteration(work_dir)


def find_next_iteration(work_dir: str) -> int:
    root = Path(work_dir)
    highest = 0
    for path in root.glob("iter_*"):
        if not path.is_dir():
            continue
        try:
            number = int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        highest = max(highest, number)
    return highest + 1


def load_pipeline_summary(work_dir: str) -> list:
    summary_path = Path(work_dir) / "pipeline_summary.json"
    if not summary_path.exists():
        return []
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def apply_preset(args: argparse.Namespace, argv: list[str]) -> None:
    if args.preset is None:
        return

    presets = {
        "debug": {
            "games": 2,
            "simulations": 4,
            "mcts_batch_size": 4,
            "eval_games": 2,
            "eval_simulations": 4,
            "eval_mcts_batch_size": 4,
            "max_plies": 48,
            "temperature_drop_ply": 12,
            "temperature_final": 0.2,
            "self_play_workers": 2,
            "epochs": 1,
            "batch_size": 32,
            "channels": 16,
            "blocks": 1,
            "replay_window": 2,
            "max_replay_positions": 2000,
        },
        "fast": {
            "games": 40,
            "simulations": 16,
            "mcts_batch_size": 8,
            "eval_games": 4,
            "eval_simulations": 16,
            "eval_mcts_batch_size": 8,
            "eval_interval": 2,
            "max_plies": 96,
            "temperature_drop_ply": 24,
            "temperature_final": 0.15,
            "self_play_workers": 2,
            "epochs": 10,
            "batch_size": 128,
            "channels": 32,
            "blocks": 2,
            "replay_window": 4,
            "max_replay_positions": 30000,
        },
        "balanced": {
            "games": 100,
            "simulations": 64,
            "mcts_batch_size": 16,
            "eval_games": 12,
            "eval_simulations": 64,
            "eval_mcts_batch_size": 16,
            "max_plies": 160,
            "temperature_drop_ply": 30,
            "temperature_final": 0.1,
            "self_play_workers": 2,
            "epochs": 3,
            "batch_size": 128,
            "channels": 64,
            "blocks": 4,
            "replay_window": 5,
            "max_replay_positions": 50000,
        },
        "strong": {
            "games": 300,
            "simulations": 128,
            "mcts_batch_size": 32,
            "eval_games": 24,
            "eval_simulations": 128,
            "eval_mcts_batch_size": 32,
            "max_plies": 200,
            "temperature_drop_ply": 40,
            "temperature_final": 0.05,
            "self_play_workers": 2,
            "epochs": 5,
            "batch_size": 256,
            "channels": 96,
            "blocks": 6,
            "replay_window": 8,
            "max_replay_positions": 200000,
        },
    }
    provided = {
        arg.split("=", 1)[0].replace("-", "_").lstrip("_")
        for arg in argv
        if arg.startswith("--")
    }
    for key, value in presets[args.preset].items():
        if key not in provided:
            setattr(args, key, value)


if __name__ == "__main__":
    main()
