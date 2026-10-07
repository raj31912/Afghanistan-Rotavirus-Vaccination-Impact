#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from rotavirus_impact.source_audit import run_audit, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mics", required=True)
    p.add_argument("--population", required=True)
    p.add_argument("--univac", required=True)
    p.add_argument("--output", default="outputs/audit/source_audit.json")
    a = p.parse_args()
    result = run_audit(a.mics, a.population, a.univac)
    out = ROOT / a.output
    write_json(result, out)
    print(f"Audit written to {out}")
    if not result["mics"]["all_semantic_checks_pass"]:
        raise SystemExit("MICS semantic audit failed; analysis halted.")


if __name__ == "__main__":
    main()
