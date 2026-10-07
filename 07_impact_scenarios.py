#!/usr/bin/env python3
"""Phase D: deterministic primary S1-S8 mortality impact and decomposition."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rotavirus_impact.impact_model import (  # noqa: E402
    INJECTABLE_NEXTGEN,
    ORAL_ROTARIX,
    RV1_TARGET_DAYS,
    RV2_TARGET_DAYS,
    gini,
    impact_fraction,
    load_curve_shapes,
    lorenz_points,
    mean_timing_shape,
    step_timing_shape,
)

CURVE_DIR = ROOT / "outputs" / "coverage_curves"
MORTALITY_PATH = ROOT / "outputs" / "mortality" / "provincial_analytic_mortality_weights_33.csv"
OUT = ROOT / "outputs" / "impact"
OUT.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "S1": "Observed delivery: province-specific coverage and timing",
    "S2": "Top-8 timing shape, province current final coverage",
    "S3": "Coverage lifted to top-8 mean, province current timing",
    "S4": "Top-8 timing plus top-8 mean coverage",
    "S5": "100% target-age oral-vaccine ceiling",
    "S6": "Hypothetical parenteral next-generation vaccine (94% non-waning) at current delivery",
    "S7": "Hypothetical parenteral next-generation vaccine (94% non-waning) with 100% on-time delivery",
    "S8": "Current final coverage, all doses exactly at 6/10 weeks",
}


def main() -> None:
    fits = pd.read_csv(CURVE_DIR / "coverage_curve_fits.csv")
    curves = pd.read_csv(CURVE_DIR / "coverage_curves_daily.csv")
    top8 = pd.read_csv(CURVE_DIR / "top8_rv1_26week_provinces.csv")["province"].tolist()
    mortality = pd.read_csv(MORTALITY_PATH)

    analytic_provinces = sorted(mortality["province"].tolist())
    if len(analytic_provinces) != 33 or "UROZGAN" in analytic_provinces:
        raise AssertionError("Scenario analysis must contain 33 provinces and exclude UROZGAN")

    shapes = load_curve_shapes(curves, fits)
    top_shape_rv1 = mean_timing_shape(shapes, top8, "RV1")
    top_shape_rv2 = mean_timing_shape(shapes, top8, "RV2")
    step_rv1 = step_timing_shape(RV1_TARGET_DAYS)
    step_rv2 = step_timing_shape(RV2_TARGET_DAYS)

    f = fits[fits.province.isin(analytic_provinces)].pivot(index="province", columns="dose", values="final_crude_coverage")
    target_rv1 = float(f.loc[top8, "RV1"].mean())
    target_rv2 = float(f.loc[top8, "RV2"].mean())

    mortality_by_p = mortality.set_index("province")
    rows: list[dict[str, object]] = []

    for province in analytic_provinces:
        c1 = float(f.loc[province, "RV1"])
        c2 = float(f.loc[province, "RV2"])
        if c2 > c1 + 1e-9:
            raise AssertionError(f"Observed nested coverage violated in {province}")
        lifted1 = max(c1, target_rv1)
        lifted2 = max(c2, target_rv2)
        lifted2 = min(lifted2, lifted1)
        own1, own2 = shapes[(province, "RV1")], shapes[(province, "RV2")]

        defs = {
            "S1": (c1, c2, own1, own2, ORAL_ROTARIX),
            "S2": (c1, c2, top_shape_rv1, top_shape_rv2, ORAL_ROTARIX),
            "S3": (lifted1, lifted2, own1, own2, ORAL_ROTARIX),
            "S4": (lifted1, lifted2, top_shape_rv1, top_shape_rv2, ORAL_ROTARIX),
            "S5": (1.0, 1.0, step_rv1, step_rv2, ORAL_ROTARIX),
            "S6": (c1, c2, own1, own2, INJECTABLE_NEXTGEN),
            "S7": (1.0, 1.0, step_rv1, step_rv2, INJECTABLE_NEXTGEN),
            "S8": (c1, c2, step_rv1, step_rv2, ORAL_ROTARIX),
        }
        baseline = float(mortality_by_p.loc[province, "baseline_deaths_composite"])
        for scenario, (s_c1, s_c2, sh1, sh2, product) in defs.items():
            frac = impact_fraction(
                rv1_final=s_c1, rv2_final=s_c2,
                rv1_timing_shape=sh1, rv2_timing_shape=sh2,
                product=product, age_distribution="burr_primary",
            )
            averted = baseline * frac
            rows.append({
                "province": province,
                "scenario": scenario,
                "scenario_description": SCENARIOS[scenario],
                "rv1_final_coverage": s_c1,
                "rv2_final_coverage": s_c2,
                "product": product.name,
                "baseline_deaths": baseline,
                "impact_fraction": frac,
                "deaths_averted": averted,
                "residual_deaths": baseline - averted,
            })

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "provincial_primary_s1_s8.csv", index=False)

    national = results.groupby("scenario", as_index=False).agg(
        baseline_deaths=("baseline_deaths", "sum"),
        deaths_averted=("deaths_averted", "sum"),
        residual_deaths=("residual_deaths", "sum"),
    )
    national["impact_fraction"] = national.deaths_averted / national.baseline_deaths
    national["scenario_description"] = national.scenario.map(SCENARIOS)
    national.to_csv(OUT / "national_primary_s1_s8.csv", index=False)

    n = national.set_index("scenario")
    decomposition = pd.DataFrame([
        {"component": "timing_exact_target_at_current_coverage", "contrast": "S8-S1", "incremental_deaths_averted": n.loc["S8", "deaths_averted"] - n.loc["S1", "deaths_averted"]},
        {"component": "timing_top8_shape_at_current_coverage", "contrast": "S2-S1", "incremental_deaths_averted": n.loc["S2", "deaths_averted"] - n.loc["S1", "deaths_averted"]},
        {"component": "coverage_to_top8_mean", "contrast": "S3-S1", "incremental_deaths_averted": n.loc["S3", "deaths_averted"] - n.loc["S1", "deaths_averted"]},
        {"component": "program_top8_timing_plus_coverage", "contrast": "S4-S1", "incremental_deaths_averted": n.loc["S4", "deaths_averted"] - n.loc["S1", "deaths_averted"]},
        {"component": "product_performance_94pct_no_waning", "contrast": "S6-S1", "incremental_deaths_averted": n.loc["S6", "deaths_averted"] - n.loc["S1", "deaths_averted"]},
        {"component": "oral_theoretical_headroom", "contrast": "S5-S1", "incremental_deaths_averted": n.loc["S5", "deaths_averted"] - n.loc["S1", "deaths_averted"]},
        {"component": "absolute_injectable_headroom", "contrast": "S7-S1", "incremental_deaths_averted": n.loc["S7", "deaths_averted"] - n.loc["S1", "deaths_averted"]},
    ])
    decomposition["percent_of_baseline_deaths"] = decomposition.incremental_deaths_averted / float(n.loc["S1", "baseline_deaths"]) * 100
    decomposition.to_csv(OUT / "national_decomposition_contrasts.csv", index=False)

    concentration_rows = []
    for scenario in ["BASELINE", "S1", "S4", "S6"]:
        if scenario == "BASELINE":
            values = mortality["baseline_deaths_composite"].to_numpy()
            metric = "baseline_deaths"
        else:
            values = results.loc[results.scenario.eq(scenario), "residual_deaths"].to_numpy()
            metric = "residual_deaths"
        concentration_rows.append({"scenario": scenario, "metric": metric, "gini": gini(values)})
        lp = lorenz_points(values)
        lp.insert(0, "scenario", scenario)
        lp.to_csv(OUT / f"lorenz_{scenario.lower()}.csv", index=False)
    pd.DataFrame(concentration_rows).to_csv(OUT / "burden_concentration_gini.csv", index=False)

    # Hard model identities / sanity checks.
    checks = {
        "province_count": int(results.province.nunique()),
        "baseline_sum": float(n.loc["S1", "baseline_deaths"]),
        "top8_rv1_final_target": target_rv1,
        "top8_rv2_final_target": target_rv2,
        "all_deaths_averted_nonnegative": bool((results.deaths_averted >= -1e-9).all()),
        "all_residual_deaths_nonnegative": bool((results.residual_deaths >= -1e-9).all()),
        "S7_ge_S6_national": bool(n.loc["S7", "deaths_averted"] >= n.loc["S6", "deaths_averted"]),
        "S7_ge_S5_national": bool(n.loc["S7", "deaths_averted"] >= n.loc["S5", "deaths_averted"]),
        "S5_ge_S1_national": bool(n.loc["S5", "deaths_averted"] >= n.loc["S1", "deaths_averted"]),
        "S8_coverage_identical_to_S1": bool(np.allclose(
            results[results.scenario.eq("S8")].sort_values("province")[["rv1_final_coverage","rv2_final_coverage"]].to_numpy(),
            results[results.scenario.eq("S1")].sort_values("province")[["rv1_final_coverage","rv2_final_coverage"]].to_numpy(),
        )),
        "oral_sequential_rv1_to_rv2_protection": True,
        "oral_one_dose_initial_ve": ORAL_ROTARIX.one_dose_peak,
        "oral_two_dose_initial_ve": ORAL_ROTARIX.two_dose_peak,
        "oral_waning_mean_months": 10.0,
        "oral_waning_shape_alpha": 3.0,
        "primary_biological_profile": "Anwari et al. 2025 Afghanistan-specific profile",
        "parenteral_hypothetical_peak": INJECTABLE_NEXTGEN.two_dose_peak,
    }
    if not all([checks["all_deaths_averted_nonnegative"], checks["all_residual_deaths_nonnegative"], checks["S7_ge_S6_national"], checks["S7_ge_S5_national"], checks["S5_ge_S1_national"], checks["S8_coverage_identical_to_S1"]]):
        raise AssertionError(f"Impact model sanity check failed: {checks}")
    (OUT / "primary_impact_audit.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    print(national.sort_values("scenario").to_string(index=False))
    print("\nDecomposition:\n", decomposition.to_string(index=False))
    print("\nAudit:\n", json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
