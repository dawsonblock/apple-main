"""Convenience wrapper for Llama 3.2-shaped benchmark runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PRESETS = ("llama32-1b", "llama32-3b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preset Llama 3.2-shaped RFSN benchmark sweeps")
    parser.add_argument("preset", choices=[*PRESETS, "all"], help="Which preset to run")
    parser.add_argument("--long-sweep", action="store_true", help="Use a longer 2k-16k sequence sweep")
    parser.add_argument("--include-full", action="store_true", help="Include full warm reconstruction alongside blockwise")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch instead of resuming an existing CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--sequence-lengths", nargs="+", type=int)
    parser.add_argument("--modes", nargs="+", choices=["pq", "hybrid"], default=["pq", "hybrid"])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-tokens", type=int, default=512)
    parser.add_argument("--warm-block-size", type=int, default=512)
    parser.add_argument("--warm-selection-blocks", type=int, default=2)
    parser.add_argument("--hot-capacity", type=int, default=1024)
    parser.add_argument("--warm-capacity", type=int, default=16384)
    parser.add_argument("--cold-capacity", type=int, default=2000000)
    parser.add_argument("--allow-cold-spill", action="store_true")
    return parser.parse_args()


def default_sequence_lengths(args: argparse.Namespace) -> list[int]:
    if args.sequence_lengths:
        return args.sequence_lengths
    if args.long_sweep:
        return [2048, 4096, 8192, 12288, 16384]
    return [2048, 4096, 8192]


def warm_read_modes(args: argparse.Namespace) -> list[str]:
    if args.include_full:
        return ["full", "blockwise"]
    return ["blockwise"]


def output_path(args: argparse.Namespace, preset: str) -> Path:
    suffix = "long" if args.long_sweep else "default"
    return args.output_dir / f"rfsn_v10_{preset}_{suffix}.csv"


def build_command(args: argparse.Namespace, preset: str) -> list[str]:
    cmd = [
        sys.executable,
        "rfsn_v10_eval_benchmark.py",
        "--preset",
        preset,
        "--sequence-lengths",
        *[str(value) for value in default_sequence_lengths(args)],
        "--modes",
        *args.modes,
        "--warm-read-modes",
        *warm_read_modes(args),
        "--warm-selection-policies",
        "all",
        "recent",
        "--trials",
        str(args.trials),
        "--seed",
        str(args.seed),
        "--step-tokens",
        str(args.step_tokens),
        "--warm-block-size",
        str(args.warm_block_size),
        "--warm-selection-blocks",
        str(args.warm_selection_blocks),
        "--hot-capacity",
        str(args.hot_capacity),
        "--warm-capacity",
        str(args.warm_capacity),
        "--cold-capacity",
        str(args.cold_capacity),
        "--output",
        str(output_path(args, preset)),
    ]
    if args.allow_cold_spill:
        cmd.append("--allow-cold-spill")
    if not args.no_resume:
        cmd.append("--resume")
    return cmd


def run_preset(args: argparse.Namespace, preset: str) -> None:
    cmd = build_command(args, preset)
    print(f"running {preset}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"wrote {output_path(args, preset)}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    presets = PRESETS if args.preset == "all" else (args.preset,)
    for preset in presets:
        run_preset(args, preset)


if __name__ == "__main__":
    main()