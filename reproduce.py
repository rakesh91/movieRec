"""Run the complete MovieLens experiment and regenerate derived artifacts."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "ml-1m"
RESULTS_DIR = ROOT / "results"

STAGES = {
    "baseline": ([sys.executable, "main.py", "--batch"], RESULTS_DIR / "metrics.json"),
    "sensitivity": (
        [sys.executable, "sensitivity_main.py"],
        RESULTS_DIR / "regularization_sensitivity",
    ),
    "genre": ([sys.executable, "genre_main.py"], RESULTS_DIR / "genre_based"),
    "hybrid": (
        [sys.executable, "Scripts/run_hybrid_validation_sweep.py"],
        RESULTS_DIR / "hybrid_validation_sweep",
    ),
    "random": (
        [sys.executable, "Scripts/run_randomized_analysis.py"],
        RESULTS_DIR / "randomized_split_seed42",
    ),
    "final": (
        [sys.executable, "Scripts/final_analysis.py"],
        RESULTS_DIR / "final_analysis",
    ),
    "resolve": (
        [sys.executable, "Scripts/resolve_mf_b_selection.py"],
        RESULTS_DIR / "final_analysis/mf_b_selection_resolution.json",
    ),
    "audits": (
        [sys.executable, "Scripts/final_patch_analysis.py"],
        RESULTS_DIR / "final_analysis/selection_effect_ci.json",
    ),
    "figure": (
        [sys.executable, "paper/make_updated_figure.py"],
        ROOT / "paper/split_fusion_history.pdf",
    ),
    "web": (
        [sys.executable, "web_showcase/build_site_data.py"],
        ROOT / "web_showcase/data/site_index.json",
    ),
}


def run_stage(name, command, expected, reuse):
    if reuse and expected.exists():
        print(f"[reuse] {name}: {expected.relative_to(ROOT)}")
        return
    if not reuse:
        target = expected if expected.is_dir() else expected.parent
        if expected.is_file():
            expected.unlink()
        elif target.exists() and name in {"sensitivity", "genre", "hybrid", "random", "final"}:
            shutil.rmtree(target)
    print(f"[run] {name}: {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild generated outputs")
    parser.add_argument(
        "--stage",
        choices=("all", *STAGES),
        default="all",
        help="run one stage, or all stages (default)",
    )
    args = parser.parse_args()

    required = ("ratings.dat", "movies.dat", "users.dat")
    missing = [name for name in required if not (DATA_DIR / name).exists()]
    if missing:
        raise SystemExit(
            "Missing MovieLens files: "
            + ", ".join(missing)
            + ". Run `python download_data.py` first."
        )

    names = list(STAGES) if args.stage == "all" else [args.stage]
    for name in names:
        command, expected = STAGES[name]
        run_stage(name, command, expected, reuse=not args.force)
    print("Reproduction workflow completed.")


if __name__ == "__main__":
    main()
