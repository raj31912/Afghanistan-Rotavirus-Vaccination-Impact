#!/usr/bin/env python3
"""Sensitivity of empirical delivery benchmarks to the number of top provinces."""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rotavirus_impact.impact_model import ORAL_ROTARIX, impact_fraction, load_curve_shapes, mean_timing_shape

CURVE_DIR = ROOT / "outputs" / "coverage_curves"
MORTALITY_PATH = ROOT / "outputs" / "mortality" / "provincial_analytic_mortality_weights_33.csv"
OUT = ROOT / "outputs" / "uncertainty_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    fits = pd.read_csv(CURVE_DIR / "coverage_curve_fits.csv")
    curves = pd.read_csv(CURVE_DIR / "coverage_curves_daily.csv")
    mortality = pd.read_csv(MORTALITY_PATH)
    provinces = sorted(mortality["province"].tolist())
    shapes = load_curve_shapes(curves, fits)
    f = fits[fits.province.isin(provinces)].pivot(index="province", columns="dose", values="final_crude_coverage")
    rv1_26 = fits[(fits.province.isin(provinces)) & fits.dose.eq("RV1")].set_index("province")["coverage_day_182"]
    ranking = rv1_26.sort_values(ascending=False).index.tolist()
    baseline = mortality.set_index("province")["baseline_deaths_composite"]

    rows = []
    membership = []
    for n_top in (5, 8, 10):
        top = ranking[:n_top]
        for rank, province in enumerate(top, 1):
            membership.append({"benchmark_size": n_top, "rank": rank, "province": province, "rv1_coverage_by_26_weeks": rv1_26.loc[province]})
        top1 = mean_timing_shape(shapes, top, "RV1")
        top2 = mean_timing_shape(shapes, top, "RV2")
        target1 = float(f.loc[top, "RV1"].mean())
        target2 = float(f.loc[top, "RV2"].mean())
        totals = {"S1": 0.0, "S2": 0.0, "S3": 0.0, "S4": 0.0}
        for p in provinces:
            c1, c2 = float(f.loc[p, "RV1"]), float(f.loc[p, "RV2"])
            own1, own2 = shapes[(p, "RV1")], shapes[(p, "RV2")]
            lifted1 = max(c1, target1)
            lifted2 = min(max(c2, target2), lifted1)
            defs = {
                "S1": (c1, c2, own1, own2),
                "S2": (c1, c2, top1, top2),
                "S3": (lifted1, lifted2, own1, own2),
                "S4": (lifted1, lifted2, top1, top2),
            }
            for scenario, (x1, x2, sh1, sh2) in defs.items():
                frac = impact_fraction(rv1_final=x1, rv2_final=x2, rv1_timing_shape=sh1, rv2_timing_shape=sh2, product=ORAL_ROTARIX, age_distribution="burr_primary")
                totals[scenario] += float(baseline.loc[p]) * frac
        timing = totals["S2"] - totals["S1"]
        coverage = totals["S3"] - totals["S1"]
        combined = totals["S4"] - totals["S1"]
        rows.append({
            "benchmark_size": n_top,
            "rv1_final_target": target1,
            "rv2_final_target": target2,
            "s1_deaths_averted": totals["S1"],
            "timing_gain_s2_minus_s1": timing,
            "coverage_gain_s3_minus_s1": coverage,
            "combined_gain_s4_minus_s1": combined,
            "coverage_to_timing_ratio": coverage / timing,
        })
    pd.DataFrame(rows).to_csv(OUT / "benchmark_size_sensitivity.csv", index=False)
    pd.DataFrame(membership).to_csv(OUT / "benchmark_size_membership.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
