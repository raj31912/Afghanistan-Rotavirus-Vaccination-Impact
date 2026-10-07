#!/usr/bin/env python3
"""Like-for-like 33-province national-average comparator.

The comparator averages current vaccination delivery across the same 33 analytic
provinces using under-five population weights, then applies that common delivery
profile to the fixed national mortality total. This removes geographic covariance
between delivery and model-assigned mortality risk without reintroducing Urozgan.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from rotavirus_impact.config import REFERENCE_RV_DEATHS_U5, SEED

IMPACT = ROOT / "outputs" / "impact"
MORT = ROOT / "outputs" / "mortality"
UNC = ROOT / "outputs" / "uncertainty_sensitivity"
OUT = ROOT / "outputs" / "reporting"
OUT.mkdir(parents=True, exist_ok=True)
N = 500


def pert(rng, lo, mode, hi, n, lam=4.0):
    if hi <= lo:
        return np.full(n, mode, dtype=float)
    a = 1 + lam * (mode - lo) / (hi - lo)
    b = 1 + lam * (hi - mode) / (hi - lo)
    return lo + (hi - lo) * rng.beta(a, b, size=n)


def summary(name, values, point):
    x = np.asarray(values, float)
    return {
        "quantity": name,
        "point": float(point),
        "mc_mean": float(x.mean()),
        "mc_median": float(np.median(x)),
        "lower_95": float(np.quantile(x, 0.025)),
        "upper_95": float(np.quantile(x, 0.975)),
    }


def main():
    det = pd.read_csv(IMPACT / "provincial_primary_s1_s8.csv")
    s1 = det[det.scenario.eq("S1")].copy()
    mort = pd.read_csv(MORT / "provincial_analytic_mortality_weights_33.csv")
    pop = mort.set_index("province")["u5_population"]
    weights = pop / pop.sum()
    point_fraction = float((s1.set_index("province")["impact_fraction"] * weights).sum())
    point_national = REFERENCE_RV_DEATHS_U5 * point_fraction

    schemes = {
        "uniform": "baseline_deaths_uniform",
        "linear_inverse_primary": "baseline_deaths_composite",
        "log_linear": "baseline_deaths_composite_log_linear",
        "bounded_2x": "baseline_deaths_composite_bounded_2x",
        "bounded_5x": "baseline_deaths_composite_bounded_5x",
        "ors_adjusted": "baseline_deaths_ors_adjusted",
        "care_seeking_adjusted": "baseline_deaths_care_seeking_adjusted",
        "underweight_adjusted": "baseline_deaths_underweight_adjusted",
    }
    frac_by_p = s1.set_index("province")["impact_fraction"]
    rows = []
    for name, col in schemes.items():
        prov = float((mort.set_index("province")[col] * frac_by_p).sum())
        rows.append({
            "mortality_weighting": name,
            "national_average_model_deaths_averted": point_national,
            "provincial_disaggregated_deaths_averted": prov,
            "national_minus_provincial": point_national - prov,
            "relative_difference_vs_provincial_pct": 100 * (point_national - prov) / prov,
        })
    pd.DataFrame(rows).to_csv(OUT / "national_vs_subnational_same_scope_impact.csv", index=False)

    # Source-aligned paired uncertainty using the same curve-draw selections and
    # PERT inputs as the primary uncertainty analysis.
    lib = pd.read_csv(UNC / "curve_fraction_library.csv")
    sel = pd.read_csv(UNC / "curve_draw_selection.csv")
    x = sel.merge(lib, on="curve_draw", how="left", validate="many_to_many")
    x = x[x.scenario.eq("S1")].copy()
    x["population_weight"] = x.province.map(weights)
    rng = np.random.default_rng(SEED + 1616)
    mort_rate = pert(rng, 22.0, 26.0, 30.0, N)
    mort_scale = mort_rate / 26.0
    oral_ve = pert(rng, 0.45, 1.0, 1.0, N)
    x["ve"] = x.replicate.map(pd.Series(oral_ve, index=np.arange(N)))
    x["impact_fraction"] = np.clip(x.curve_impact_fraction * x.ve, 0, 1)
    avg_frac = x.groupby("replicate").apply(
        lambda g: float((g.impact_fraction * g.population_weight).sum()), include_groups=False
    ).sort_index().to_numpy()
    natavg = REFERENCE_RV_DEATHS_U5 * mort_scale * avg_frac
    prov_draws = pd.read_csv(UNC / "source_aligned_national_scenario_draws.csv").sort_values("replicate")["S1"].to_numpy(float)
    diff = natavg - prov_draws
    rel = 100 * diff / prov_draws
    paired = pd.DataFrame({
        "replicate": np.arange(N),
        "national_average_deaths_averted": natavg,
        "provincial_disaggregated_deaths_averted": prov_draws,
        "difference": diff,
        "relative_difference_pct": rel,
    })
    paired.to_csv(OUT / "source_aligned_same_scope_national_vs_subnational_draws.csv", index=False)
    point_prov = float(s1.deaths_averted.sum())
    summ = pd.DataFrame([
        summary("national_average_deaths_averted", natavg, point_national),
        summary("provincial_disaggregated_deaths_averted", prov_draws, point_prov),
        summary("national_minus_provincial_deaths_averted", diff, point_national - point_prov),
        summary("national_relative_difference_pct", rel, 100 * (point_national - point_prov) / point_prov),
    ])
    summ.to_csv(OUT / "source_aligned_same_scope_national_vs_subnational_uncertainty.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n", summ.to_string(index=False))


if __name__ == "__main__":
    main()
