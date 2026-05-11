from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def collect_selfplay_files(work_dir: str, window: int) -> list[Path]:
    root = Path(work_dir)
    files = sorted(root.glob("iter_*/selfplay.jsonl"))
    if window > 0:
        files = files[-window:]
    return files


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def build_replay_buffer(
    files: list[Path],
    out: str,
    max_positions: int = 0,
    seed: int | None = None,
) -> dict:
    if seed is not None:
        random.seed(seed)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(files) == 1 and max_positions <= 0:
        shutil.copy2(files[0], out_path)
        return {
            "out": str(out_path),
            "files": [str(files[0])],
            "positions": count_lines(out_path),
            "sampled": False,
        }

    reservoir: list[str] = []
    seen = 0
    with out_path.open("w", encoding="utf-8") as output:
        for path in files:
            with path.open(encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    seen += 1
                    if max_positions <= 0:
                        output.write(line)
                    elif len(reservoir) < max_positions:
                        reservoir.append(line)
                    else:
                        index = random.randint(0, seen - 1)
                        if index < max_positions:
                            reservoir[index] = line

        if max_positions > 0:
            random.shuffle(reservoir)
            output.writelines(reservoir)

    return {
        "out": str(out_path),
        "files": [str(path) for path in files],
        "positions": min(seen, max_positions) if max_positions > 0 else seen,
        "seen": seen,
        "sampled": max_positions > 0 and seen > max_positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="runs")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--out", default="data/replay.jsonl")
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    files = collect_selfplay_files(args.work_dir, args.window)
    if not files:
        raise SystemExit(f"No self-play files found under {args.work_dir}")

    result = build_replay_buffer(
        files=files,
        out=args.out,
        max_positions=args.max_positions,
        seed=args.seed,
    )
    print(result)


if __name__ == "__main__":
    main()

