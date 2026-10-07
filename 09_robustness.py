#!/usr/bin/env python3
"""Phase F: COVID-era birth-cohort and 15-Aug-2021 governance robustness checks."""
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
from rotavirus_impact.mics_reconstruction import analytic_sample, reconstruct_rotavirus_coverage  # noqa: E402
from rotavirus_impact.robustness import (  # noqa: E402
    add_bootstrap_ci,
    add_gregorian_birth_date,
    bootstrap_two_group_difference,
    grouped_coverage,
)
from rotavirus_impact.sav_numeric import read_numeric_subset  # noqa: E402

MICS = raw_input_path("mics")
OUT = ROOT / "outputs" / "robustness"
OUT.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "HH1", "HH2", "LN", "CAGE", "CAGED", "HH7", "CHWEIGHT", "PSU", "STRATUM",
    "UF7D_G", "UF7M_G", "UF7Y_G",
    "UB1D", "UB1M", "UB1Y",
    "IM6R1D", "IM6R1M", "IM6R1Y", "IM6R2D", "IM6R2M", "IM6R2Y", "IM24", "IM25",
]

COHORT_LABELS = [
    "Oct-Nov 2020",
    "Dec 2020-Jan 2021",
    "Feb-Mar 2021",
    "Apr-May 2021",
    "Jun-Jul 2021",
    "Aug-Sep 2021",
    "Oct-Nov 2021",
    "Dec 2021",
]
COHORT_BOUNDS = pd.to_datetime([
    "2020-10-01", "2020-12-01", "2021-02-01", "2021-04-01", "2021-06-01",
    "2021-08-01", "2021-10-01", "2021-12-01", "2022-01-01",
])


def main() -> None:
    df = reconstruct_rotavirus_coverage(analytic_sample(read_numeric_subset(MICS, FIELDS)))
    df = add_gregorian_birth_date(df)

    # Independent birth-date source cross-check.
    d = pd.to_numeric(df["birth_date_crosscheck_difference_days"], errors="coerce").dropna()
    birth_audit = {
        "analytic_n": int(len(df)),
        "birth_date_primary_shamsi_exact_n": int(df.birth_date_method.eq("shamsi_exact_birth_date").sum()),
        "birth_date_interview_age_fallback_n": int(df.birth_date_method.eq("interview_minus_exact_age_fallback").sum()),
        "birth_date_unresolved_n": int(df.birth_date_method.eq("unresolved").sum()),
        "independent_birth_date_crosscheck_n": int(len(d)),
        "crosscheck_difference_days_counts": {str(int(k)): int(v) for k, v in d.value_counts().sort_index().items()},
        "crosscheck_max_absolute_days": float(d.abs().max()) if len(d) else None,
    }
    if len(d) and d.abs().max() > 1:
        raise AssertionError("Independent birth-date derivations differ by more than one day")

    valid_birth = df[df.birth_date_gregorian.notna()].copy()

    # COVID-era two-month birth cohorts, exactly the requested Oct-2020 to Dec-2021 window.
    covid = valid_birth[
        valid_birth.birth_date_gregorian.between(pd.Timestamp("2020-10-01"), pd.Timestamp("2021-12-31"))
    ].copy()
    covid["birth_cohort_2m"] = pd.cut(
        covid.birth_date_gregorian,
        bins=COHORT_BOUNDS,
        labels=COHORT_LABELS,
        right=False,
        include_lowest=True,
    )
    covid = covid[covid.birth_cohort_2m.notna()].copy()
    covid["birth_cohort_2m"] = covid.birth_cohort_2m.astype(str)
    cohort = grouped_coverage(covid, "birth_cohort_2m")
    cohort["_order"] = cohort.birth_cohort_2m.map({x: i for i, x in enumerate(COHORT_LABELS)})
    cohort = cohort.sort_values("_order").drop(columns="_order")
    cohort = add_bootstrap_ci(
        cohort, covid, group_col="birth_cohort_2m", outcome_col="rv1_vaccinated",
        estimate_col="rv1_coverage", replicates=1000, seed=SEED + 10,
    )
    cohort = add_bootstrap_ci(
        cohort, covid, group_col="birth_cohort_2m", outcome_col="rv2_vaccinated",
        estimate_col="rv2_coverage", replicates=1000, seed=SEED + 20,
    )
    cohort.to_csv(OUT / "covid_two_month_birth_cohort_coverage.csv", index=False)

    # Exact age-at-interview cross-tabulation. It is deliberately
    # descriptive because birth cohort and interview age are strongly linked by the six-month field period.
    cross_rows = []
    for (cohort_name, age_month), g in covid.groupby(["birth_cohort_2m", "CAGE"], sort=False):
        cross_rows.append({
            "birth_cohort_2m": cohort_name,
            "age_at_interview_months": int(age_month),
            "n": int(len(g)),
            "weighted_sample_mass": float(g.CHWEIGHT.sum()),
            "rv1_coverage": float(np.average(g.rv1_vaccinated.astype(float), weights=g.CHWEIGHT)),
            "rv2_coverage": float(np.average(g.rv2_vaccinated.astype(float), weights=g.CHWEIGHT)),
        })
    age_cross = pd.DataFrame(cross_rows)
    age_cross.to_csv(OUT / "covid_birth_cohort_by_age_at_interview_crosstab.csv", index=False)

    # Compact age bands for human-readable supplementary reporting.
    covid["age_band_interview"] = pd.cut(
        covid.CAGE,
        bins=[11, 14, 17, 20, 23],
        labels=["12-14", "15-17", "18-20", "21-23"],
        include_lowest=True,
    ).astype(str)
    band_rows = []
    for (cohort_name, age_band), g in covid.groupby(["birth_cohort_2m", "age_band_interview"], sort=False):
        band_rows.append({
            "birth_cohort_2m": cohort_name,
            "age_band_interview_months": age_band,
            "n": int(len(g)),
            "weighted_sample_mass": float(g.CHWEIGHT.sum()),
            "rv1_coverage": float(np.average(g.rv1_vaccinated.astype(float), weights=g.CHWEIGHT)),
            "rv2_coverage": float(np.average(g.rv2_vaccinated.astype(float), weights=g.CHWEIGHT)),
        })
    pd.DataFrame(band_rows).to_csv(OUT / "covid_birth_cohort_by_age_band.csv", index=False)

    # Governance transition: births before vs on/after 15 Aug 2021.
    transition = pd.Timestamp("2021-08-15")
    gov = valid_birth.copy()
    gov["governance_birth_group"] = np.where(
        gov.birth_date_gregorian < transition,
        "born_before_2021_08_15",
        "born_on_or_after_2021_08_15",
    )
    gov_point = grouped_coverage(gov, "governance_birth_group")
    gov_point = add_bootstrap_ci(
        gov_point, gov, group_col="governance_birth_group", outcome_col="rv1_vaccinated",
        estimate_col="rv1_coverage", replicates=1000, seed=SEED + 30,
    )
    gov_point = add_bootstrap_ci(
        gov_point, gov, group_col="governance_birth_group", outcome_col="rv2_vaccinated",
        estimate_col="rv2_coverage", replicates=1000, seed=SEED + 40,
    )
    gov_point.to_csv(OUT / "governance_transition_coverage.csv", index=False)

    pre = "born_before_2021_08_15"
    post = "born_on_or_after_2021_08_15"
    gov_diff_rows = []
    point_idx = gov_point.set_index("governance_birth_group")
    for dose in (1, 2):
        col = f"rv{dose}_vaccinated"
        diff = bootstrap_two_group_difference(
            gov,
            group_col="governance_birth_group",
            pre_label=pre,
            post_label=post,
            outcome_col=col,
            replicates=1000,
            seed=SEED + 50 + dose,
        )
        point_difference = float(point_idx.loc[post, f"rv{dose}_coverage"] - point_idx.loc[pre, f"rv{dose}_coverage"])
        gov_diff_rows.append({
            "dose": f"RV{dose}",
            "post_minus_pre_percentage_points": point_difference * 100.0,
            "lower95_difference_percentage_points": diff["lower95_difference"] * 100.0,
            "upper95_difference_percentage_points": diff["upper95_difference"] * 100.0,
            "bootstrap_replicates": diff["bootstrap_replicates"],
        })
    gov_diff = pd.DataFrame(gov_diff_rows)
    gov_diff.to_csv(OUT / "governance_transition_difference.csv", index=False)

    # Quantify cohort-age coupling rather than pretending the cross-tab fully removes it.
    cohort_order = {x: i for i, x in enumerate(COHORT_LABELS)}
    tmp = covid.copy()
    tmp["cohort_index"] = tmp.birth_cohort_2m.map(cohort_order).astype(float)
    weighted_cov = np.cov(
        tmp.cohort_index.to_numpy(), tmp.CAGE.to_numpy(),
        aweights=tmp.CHWEIGHT.to_numpy(), ddof=0,
    )
    weighted_corr = float(weighted_cov[0, 1] / np.sqrt(weighted_cov[0, 0] * weighted_cov[1, 1]))

    audit = {
        **birth_audit,
        "covid_window_n": int(len(covid)),
        "covid_window": "2020-10-01 through 2021-12-31",
        "two_month_bins": COHORT_LABELS,
        "last_bin_is_single_month": True,
        "survey_ci_method": "1000-replicate stratified PSU bootstrap; singleton-PSU strata carried unchanged",
        "weighted_correlation_birth_cohort_index_vs_age_at_interview_months": weighted_corr,
        "governance_transition_date": "2021-08-15",
        "governance_valid_birth_n": int(len(gov)),
        "interpretation_rule": "A collapse is not claimed unless post-transition coverage is materially lower and the bootstrap interval for post-minus-pre is below zero.",
    }
    (OUT / "robustness_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("Two-month birth cohorts")
    print(cohort.to_string(index=False))
    print("\nGovernance transition")
    print(gov_point.to_string(index=False))
    print("\nGovernance differences")
    print(gov_diff.to_string(index=False))
    print("\nAudit")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
