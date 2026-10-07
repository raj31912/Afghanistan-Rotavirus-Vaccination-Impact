#!/usr/bin/env python3
"""Phase C: MICS care-access proxies and provincial baseline mortality weights."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rotavirus_impact.config import REFERENCE_RV_DEATHS_U5  # noqa: E402
from rotavirus_impact.paths import raw_input_path  # noqa: E402
from rotavirus_impact.mortality_weights import (  # noqa: E402
    FORMAL_CA6,
    MICS_REPORT_BENCHMARKS,
    PRIVATE_ALL_CA6,
    PUBLIC_FORMAL_CA6,
    add_access_scores,
    allocate_mortality,
    merge_population,
    mortality_allocation_audit,
    national_proxy_qc,
    provincial_proxies,
)
from rotavirus_impact.sav_numeric import read_dictionary, read_mixed_subset  # noqa: E402

MICS_PATH = raw_input_path("mics")
POP_PATH = raw_input_path("population")
OUT = ROOT / "outputs" / "mortality"
OUT.mkdir(parents=True, exist_ok=True)

CA6_CODES = sorted(set(FORMAL_CA6 + PRIVATE_ALL_CA6 + PUBLIC_FORMAL_CA6))
FIELDS = [
    "HH7", "CHWEIGHT", "CA1", "CA5", "CA7A", "CA7B", "CA7E", "WAZ2",
] + [f"CA6{x}" for x in CA6_CODES]


def main() -> None:
    _header, _vars, labels = read_dictionary(MICS_PATH)
    df = read_mixed_subset(MICS_PATH, FIELDS)
    df["province"] = df["HH7"].map(labels["HH7"])

    national = national_proxy_qc(df)
    benchmark_checks = {}
    for key in ("ors", "formal_care", "public_care", "private_care", "underweight"):
        observed = float(national[f"{key}_percent"])
        expected = float(MICS_REPORT_BENCHMARKS[f"{key}_percent"])
        difference_pp = observed - expected
        benchmark_checks[key] = {
            "reconstructed_percent": observed,
            "published_percent": expected,
            "difference_percentage_points": difference_pp,
            "within_0_15_percentage_points": abs(difference_pp) <= 0.15,
        }
        if abs(difference_pp) > 0.15:
            raise AssertionError(f"{key} reconstruction differs from MICS report by >0.15 pp")

    proxy = provincial_proxies(df)
    access = add_access_scores(proxy, ddof=0)
    merged = merge_population(access, str(POP_PATH))
    results = allocate_mortality(merged, reference_deaths=REFERENCE_RV_DEATHS_U5)
    results.to_csv(OUT / "provincial_care_access_and_mortality_weights.csv", index=False)

    # Prespecified analytic scenario set excludes Urozgan for sparse timing data.
    # Re-normalise the remaining 33 province burden weights to the same national
    # reference total so provincial scenario sums remain directly interpretable
    # as the national analytic estimate. The 34-province allocation above is
    # retained for descriptive reporting.
    analytic33 = merged.loc[merged["province"].str.upper().ne("UROZGAN")].copy()
    analytic33 = allocate_mortality(analytic33, reference_deaths=REFERENCE_RV_DEATHS_U5)
    analytic33.to_csv(OUT / "provincial_analytic_mortality_weights_33.csv", index=False)

    national_out = {
        "source_semantic_qc": national,
        "published_benchmark_checks": benchmark_checks,
        "formal_care_definition": "CA6 A/B/C/E/F/G/H/I/J/M/O; private pharmacy K excluded",
        "ors_definition": "CA7A or CA7B or Afghanistan-specific Salamati ORS+zinc CA7E",
        "underweight_definition": "valid WHO WAZ2 in [-6,6], underweight if WAZ2 < -2",
        "standardisation": "unweighted province-level z-scores, population ddof=0",
        "composite": "(z_ORS + z_formal_care - z_underweight) / 3",
        "primary_cfr_link": "linear-inverse 1/[1 + (composite - observed minimum)]; then population-normalised to weighted mean 1",
        "alternative_cfr_links": {
            "log_linear": "exp(-composite)",
            "bounded_2x": "min-max monotone mapping with worst:best raw CFR ratio fixed at 2:1",
            "bounded_5x": "min-max monotone mapping with worst:best raw CFR ratio fixed at 5:1",
        },
        "allocation_audit_34_descriptive": mortality_allocation_audit(results, REFERENCE_RV_DEATHS_U5),
        "allocation_audit_33_analytic": mortality_allocation_audit(analytic33, REFERENCE_RV_DEATHS_U5),
        "analytic_exclusion": "UROZGAN excluded before mortality re-normalisation for scenario analysis",
    }
    (OUT / "mortality_weight_audit.json").write_text(json.dumps(national_out, indent=2), encoding="utf-8")

    ranking = results[[
        "province", "u5_population", "ors_percent", "formal_care_percent", "underweight_percent",
        "care_access_composite_z", "cfr_multiplier_composite", "baseline_deaths_composite"
    ]].sort_values("baseline_deaths_composite", ascending=False)
    ranking.to_csv(OUT / "provincial_primary_mortality_ranking.csv", index=False)

    print(json.dumps(national_out, indent=2))


if __name__ == "__main__":
    main()
