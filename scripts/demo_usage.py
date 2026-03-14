"""Demonstration script for using the BOUND-SDE dataset generator and CLI."""

import argparse
import os
import subprocess
from pathlib import Path

from scripts.generate_dataset import generate_dataset


def demo_generation(config_path: Path, output_path: Path, seed: int = 0) -> None:
    """Generate data and show paths."""

    print("\n[STEP 1] Loading configuration from", config_path)
    generated = generate_dataset(config_path, output_path, seed=seed)
    print("Generated dataset at", generated)


def demo_fit(config_path: Path, data_path: Path, output_path: Path, steps: int = 20) -> None:
    """Run main.py fit command with verbose logging."""

    print("\n[STEP 2] Fitting model using config", config_path)
    proc = subprocess.run(
        [
            "python",
            "main.py",
            "fit",
            "--data",
            str(data_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--steps",
            str(steps),
        ],
        check=True,
    )
    print("Fit completed with return code", proc.returncode)


def main(space: argparse.Namespace) -> None:
    print("Starting demo run for BOUND-SDE" )
    gen_out = Path(space.output_dir) / "demo_dataset.npz"
    fit_out = Path(space.output_dir) / "demo_params.npz"
    demo_generation(Path(space.config), gen_out, seed=space.seed)
    demo_fit(Path(space.config), gen_out, fit_out, steps=space.steps)
    print("\nDemo finished. Dataset and params saved to", space.output_dir)


def cli_entry() -> None:
    parser = argparse.ArgumentParser(description="BOUND-SDE usage demo")
    parser.add_argument("--config", default="configs/example_config.json")
    parser.add_argument("--output-dir", default="demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)


if __name__ == "__main__":
    cli_entry()
