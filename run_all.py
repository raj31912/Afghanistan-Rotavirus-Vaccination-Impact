#!/usr/bin/env python3
"""Run the complete Paper 1 analysis pipeline.

Raw inputs are passed explicitly and exposed to stage scripts through environment
variables, avoiding machine-specific paths in the repository.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

STAGES = [
    "02_mics_reconstruction.py",
    "03_timing_quality.py",
    "04_timing_imputation.py",
    "05_coverage_curves.py",
    "06_mortality_weights.py",
    "07_impact_scenarios.py",
    "08_uncertainty_sensitivity.py",
    "09_robustness.py",
    "10_face_validity.py",
    "15_one_dose_ratio_sensitivity.py",
    "16_source_aligned_uncertainty.py",
    "17_pre_rv2_protection_sensitivity.py",
    "18_benchmark_size_sensitivity.py",
    "19_same_scope_national_average.py",
    "20_risk_coverage_correlation.py",
    "11_reporting_outputs.py",
    "12_publication_figures.py",
    "13_release_manifest.py",
]


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mics", required=True, type=Path)
    p.add_argument("--population", required=True, type=Path)
    p.add_argument("--univac", required=True, type=Path)
    p.add_argument("--skip-tests", action="store_true")
    args = p.parse_args()

    for name, path in (("mics", args.mics), ("population", args.population), ("univac", args.univac)):
        if not path.exists():
            p.error(f"{name} file not found: {path}")

    env = os.environ.copy()
    env.update({
        "RVI_MICS_PATH": str(args.mics.resolve()),
        "RVI_POPULATION_PATH": str(args.population.resolve()),
        "RVI_UNIVAC_PATH": str(args.univac.resolve()),
    })

    run([
        sys.executable, "01_source_audit.py",
        "--mics", str(args.mics.resolve()),
        "--population", str(args.population.resolve()),
        "--univac", str(args.univac.resolve()),
    ], env)
    for stage in STAGES:
        run([sys.executable, stage], env)
    if not args.skip_tests:
        run([sys.executable, "-m", "pytest", "-q"], env)


if __name__ == "__main__":
    main()
