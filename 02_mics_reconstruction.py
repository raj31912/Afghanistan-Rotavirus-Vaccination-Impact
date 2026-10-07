"""Phase A: reconstruct the MICS 12-23 month analytic sample and Rotarix coverage."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from rotavirus_impact.mics_reconstruction import (  # noqa: E402
    analytic_sample,
    reconstruct_rotavirus_coverage,
    survey_weighted_mean,
)
from rotavirus_impact.sav_numeric import read_dictionary, read_numeric_subset  # noqa: E402
from rotavirus_impact.paths import raw_input_path  # noqa: E402


MICS_PATH = raw_input_path("mics")
OUT = ROOT / "outputs" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "HH1", "HH2", "LN", "CAGE", "HH6", "HH7", "CHWEIGHT", "WINDEX5",
    "UB1D", "UB1M", "UB1Y", "UB1A", "IM2", "IM3", "IM5",
    "IM6R1D", "IM6R1M", "IM6R1Y", "IM6R2D", "IM6R2M", "IM6R2Y",
    "IM24", "IM25", "PSU", "STRATUM",
]


def main() -> None:
    header, _variables, value_labels = read_dictionary(MICS_PATH)
    df = read_numeric_subset(MICS_PATH, FIELDS)
    analytic = analytic_sample(df)
    analytic = reconstruct_rotavirus_coverage(analytic)
    province_labels = value_labels["HH7"]
    analytic["province"] = analytic["HH7"].map(province_labels)

    national = {
        "raw_children_cases": int(header.case_count),
        "analytic_children_12_23_months": int(len(analytic)),
        "rv1_unweighted_n": int(analytic["rv1_vaccinated"].sum()),
        "rv2_unweighted_n": int(analytic["rv2_vaccinated"].sum()),
        "rv1_weighted_coverage": survey_weighted_mean(analytic["rv1_vaccinated"], analytic["CHWEIGHT"]),
        "rv2_weighted_coverage": survey_weighted_mean(analytic["rv2_vaccinated"], analytic["CHWEIGHT"]),
        "rv1_evidence_counts": analytic["rv1_evidence"].value_counts().to_dict(),
        "rv2_evidence_counts": analytic["rv2_evidence"].value_counts().to_dict(),
    }

    provincial_rows = []
    for province, group in analytic.groupby("province", sort=True):
        provincial_rows.append(
            {
                "province": province,
                "n_12_23_months": int(len(group)),
                "rv1_n": int(group["rv1_vaccinated"].sum()),
                "rv2_n": int(group["rv2_vaccinated"].sum()),
                "rv1_weighted_coverage": survey_weighted_mean(group["rv1_vaccinated"], group["CHWEIGHT"]),
                "rv2_weighted_coverage": survey_weighted_mean(group["rv2_vaccinated"], group["CHWEIGHT"]),
                "rv1_complete_written_date_n": int((group["rv1_evidence"] == "complete_written_date").sum()),
                "rv2_complete_written_date_n": int((group["rv2_evidence"] == "complete_written_date").sum()),
            }
        )

    (OUT / "mics_national_reconstruction.json").write_text(json.dumps(national, indent=2), encoding="utf-8")
    import pandas as pd
    pd.DataFrame(provincial_rows).to_csv(OUT / "mics_provincial_coverage_audit.csv", index=False)

    print(json.dumps(national, indent=2))


if __name__ == "__main__":
    main()
