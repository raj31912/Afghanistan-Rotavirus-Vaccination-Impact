#!/usr/bin/env python3
"""Phase B2: empirical KM curves and province-specific Clark 5-parameter fits."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rotavirus_impact.config import SEED  # noqa: E402
from rotavirus_impact.paths import raw_input_path  # noqa: E402
from rotavirus_impact.coverage_curves import (  # noqa: E402
    CURVE_ENDPOINT_DAYS,
    clark_raw_coverage,
    fit_clark_curve,
    rescaled_coverage,
    timing_metrics,
    validate_against_clark_examples,
    validate_against_clark_zambia_dtp3,
    weighted_km_failure,
)
from rotavirus_impact.mics_reconstruction import analytic_sample, reconstruct_rotavirus_coverage, survey_weighted_mean  # noqa: E402
from rotavirus_impact.sav_numeric import read_dictionary, read_numeric_subset  # noqa: E402
from rotavirus_impact.timing_imputation import impute_vaccination_ages  # noqa: E402

MICS_PATH = raw_input_path("mics")
OUT = ROOT / "outputs" / "coverage_curves"
OUT.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "HH1", "HH2", "LN", "CAGE", "CAGED", "HH6", "HH7", "CHWEIGHT", "WINDEX5",
    "UB1D", "UB1M", "UB1Y",
    "IM6R1D", "IM6R1M", "IM6R1Y", "IM6R2D", "IM6R2M", "IM6R2Y",
    "IM24", "IM25", "PSU", "STRATUM",
]
TARGET = {1: 42, 2: 70}
CHECK_DAYS = [42, 70, 98, 182, 365, 546, 728]


def fit_one(group: pd.DataFrame, *, province: str, dose: int, seed: int, initial_params=None, starts=5):
    crude = survey_weighted_mean(group[f"rv{dose}_vaccinated"], group["CHWEIGHT"])
    km = weighted_km_failure(
        group,
        event_age_col=f"rv{dose}_age_days",
        vaccinated_col=f"rv{dose}_vaccinated",
        max_day=CURVE_ENDPOINT_DAYS,
    )
    fit = fit_clark_curve(km, TARGET[dose], seed=seed, starts=starts, initial_params=initial_params)
    params = fit.params
    raw_endpoint = float(clark_raw_coverage(np.array([CURVE_ENDPOINT_DAYS]), TARGET[dose], params)[0])

    row = {
        "province": province,
        "dose": f"RV{dose}",
        "target_days": TARGET[dose],
        "n_children": int(len(group)),
        "vaccinated_n": int(group[f"rv{dose}_vaccinated"].sum()),
        "final_crude_coverage": crude,
        "p1": fit.p1,
        "p2": fit.p2,
        "p3": fit.p3,
        "p4": fit.p4,
        "p5": fit.p5,
        "rmse": fit.rmse,
        "nfev": fit.nfev,
        "fit_success": fit.success,
        "raw_fit_at_104w": raw_endpoint,
    }
    se = fit.standard_errors
    for i in range(5):
        row[f"se_p{i+1}"] = float(se[i])
    cov = fit.covariance
    for i in range(5):
        for j in range(5):
            row[f"cov_p{i+1}_p{j+1}"] = float(cov[i, j]) if cov is not None else float("nan")
    for day in CHECK_DAYS:
        row[f"coverage_day_{day}"] = float(rescaled_coverage(
            np.array([day]), target_days=TARGET[dose], params=params,
            final_coverage=crude, endpoint_days=CURVE_ENDPOINT_DAYS,
        )[0])
    row.update({f"timing_{k}": v for k, v in timing_metrics(group, dose, TARGET[dose]).items()})

    km = km.assign(province=province, dose=f"RV{dose}")
    km["fitted_raw"] = clark_raw_coverage(km["age_days"].to_numpy(), TARGET[dose], params)
    km["fitted_rescaled"] = rescaled_coverage(
        km["age_days"].to_numpy(), target_days=TARGET[dose], params=params,
        final_coverage=crude, endpoint_days=CURVE_ENDPOINT_DAYS,
    )
    return row, km


def main() -> None:
    _header, _vars, labels = read_dictionary(MICS_PATH)
    df = reconstruct_rotavirus_coverage(analytic_sample(read_numeric_subset(MICS_PATH, FIELDS)))
    df["province"] = df["HH7"].map(labels["HH7"])
    df = impute_vaccination_ages(df, seed=SEED)

    formula_validation = validate_against_clark_examples()
    if not formula_validation["all_six_checks_pass"]:
        raise AssertionError("Clark mathematical form failed published Appendix Table S15 validation")

    fit_rows = []
    curve_frames = []

    # National fits serve as an independent overall benchmark.
    national_params = {}
    for dose in (1, 2):
        row, km = fit_one(
            df, province="AFGHANISTAN", dose=dose, seed=SEED + dose, starts=8
        )
        national_params[dose] = np.array([row[f"p{i}"] for i in range(1, 6)], dtype=float)
        fit_rows.append(row); curve_frames.append(km)

    for p_idx, (province, g) in enumerate(df.groupby("province", sort=True), start=1):
        for dose in (1, 2):
            row, km = fit_one(
                g, province=province, dose=dose, seed=SEED + 100 * p_idx + dose,
                initial_params=national_params[dose], starts=1
            )
            fit_rows.append(row); curve_frames.append(km)

    fits = pd.DataFrame(fit_rows)
    curves = pd.concat(curve_frames, ignore_index=True)
    fits.to_csv(OUT / "coverage_curve_fits.csv", index=False)
    curves.to_csv(OUT / "coverage_curves_daily.csv", index=False)

    # Define top 8 using fitted/rescaled RV1 coverage at 26 weeks (182 days),
    # excluding Urozgan from analytic scenario construction as specified.
    rv1 = fits[(fits.dose == "RV1") & (fits.province != "AFGHANISTAN")].copy()
    analytic = rv1[rv1.province.str.upper() != "UROZGAN"].copy()
    top8 = analytic.sort_values("coverage_day_182", ascending=False).head(8)
    top8_out = top8[["province", "coverage_day_182", "final_crude_coverage", "rmse", "vaccinated_n"]].copy()
    top8_out["rank_26week_rv1"] = np.arange(1, len(top8_out) + 1)
    top8_out.to_csv(OUT / "top8_rv1_26week_provinces.csv", index=False)

    empirical_rank_rows = []
    rv1_curves = curves[(curves.dose == "RV1") & (curves.province != "AFGHANISTAN")]
    for province, g in rv1_curves.groupby("province", sort=True):
        crude = float(rv1.loc[rv1.province == province, "final_crude_coverage"].iloc[0])
        km182 = float(g.loc[g.age_days == 182, "km_coverage"].iloc[0])
        km728 = float(g.loc[g.age_days == CURVE_ENDPOINT_DAYS, "km_coverage"].iloc[0])
        empirical = crude * km182 / km728 if km728 > 0 else 0.0
        fitted = float(rv1.loc[rv1.province == province, "coverage_day_182"].iloc[0])
        empirical_rank_rows.append({
            "province": province, "empirical_rescaled_26week": empirical,
            "fitted_26week": fitted, "absolute_difference": abs(fitted - empirical)
        })
    empirical_rank = pd.DataFrame(empirical_rank_rows)
    empirical_rank.to_csv(OUT / "rv1_26week_fit_vs_empirical.csv", index=False)
    empirical_top8 = empirical_rank[empirical_rank.province.str.upper() != "UROZGAN"].sort_values(
        "empirical_rescaled_26week", ascending=False
    ).head(8).province.tolist()

    targets = {
        "rv1_top8_mean_final_coverage": float(top8["final_crude_coverage"].mean()),
        "rv1_top8_mean_26week_coverage": float(top8["coverage_day_182"].mean()),
        "rv2_top8_mean_final_coverage_same_provinces": float(
            fits[(fits.dose == "RV2") & fits.province.isin(top8.province)]["final_crude_coverage"].mean()
        ),
        "top8_provinces": top8.province.tolist(),
        "empirical_top8_provinces": empirical_top8,
        "top8_membership_identical_fit_vs_empirical": set(empirical_top8) == set(top8.province.tolist()),
        "median_abs_fit_vs_empirical_26week_difference": float(empirical_rank.absolute_difference.median()),
        "max_abs_fit_vs_empirical_26week_difference": float(empirical_rank.absolute_difference.max()),
        "curve_endpoint_days": CURVE_ENDPOINT_DAYS,
        "clark_formula_validation": formula_validation,
        "fit_rmse_summary": {
            "national_rv1": float(fits[(fits.province == "AFGHANISTAN") & (fits.dose == "RV1")].rmse.iloc[0]),
            "national_rv2": float(fits[(fits.province == "AFGHANISTAN") & (fits.dose == "RV2")].rmse.iloc[0]),
            "provincial_median": float(fits[fits.province != "AFGHANISTAN"].rmse.median()),
            "provincial_max": float(fits[fits.province != "AFGHANISTAN"].rmse.max()),
        },
    }
    (OUT / "coverage_curve_summary.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")

    print(json.dumps(targets, indent=2))


if __name__ == "__main__":
    main()
