"""Phase B0: descriptive quality audit of complete MICS Rotarix dates."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rotavirus_impact.mics_reconstruction import analytic_sample, reconstruct_rotavirus_coverage
from rotavirus_impact.sav_numeric import read_dictionary, read_numeric_subset
from rotavirus_impact.timing_quality import add_timing_qc, timing_summary
from rotavirus_impact.paths import raw_input_path

MICS_PATH = raw_input_path("mics")
OUT = ROOT / "outputs" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "HH1", "HH2", "LN", "CAGE", "HH6", "HH7", "CHWEIGHT", "WINDEX5",
    "UB1D", "UB1M", "UB1Y",
    "IM6R1D", "IM6R1M", "IM6R1Y", "IM6R2D", "IM6R2M", "IM6R2Y",
    "IM24", "IM25",
]


def main() -> None:
    _header, _vars, labels = read_dictionary(MICS_PATH)
    df = reconstruct_rotavirus_coverage(analytic_sample(read_numeric_subset(MICS_PATH, FIELDS)))
    df["province"] = df["HH7"].map(labels["HH7"])
    qc = add_timing_qc(df)

    national = timing_summary(qc)
    (OUT / "mics_timing_quality_national.json").write_text(json.dumps(national, indent=2), encoding="utf-8")

    province_rows = []
    for province, group in qc.groupby("province", sort=True):
        summary = timing_summary(group)
        province_rows.append({
            "province": province,
            "n_12_23_months": len(group),
            "rv1_valid_date_n": summary["rv1"]["calendar_valid_age_n"],
            "rv2_valid_date_n": summary["rv2"]["calendar_valid_age_n"],
            "rv1_median_age_weeks": summary["rv1"]["median_age_weeks"],
            "rv2_median_age_weeks": summary["rv2"]["median_age_weeks"],
            "rv1_under_2_weeks_n": summary["rv1"]["under_2_weeks_n"],
            "rv1_over_52_weeks_n": summary["rv1"]["over_52_weeks_n"],
            "rv2_under_2_weeks_n": summary["rv2"]["under_2_weeks_n"],
            "rv2_over_52_weeks_n": summary["rv2"]["over_52_weeks_n"],
            "dose2_before_dose1_n": summary["dose_sequence"]["dose2_before_dose1_n"],
            "same_day_n": summary["dose_sequence"]["same_day_n"],
        })
    with (OUT / "mics_timing_quality_by_province.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(province_rows[0].keys()))
        writer.writeheader(); writer.writerows(province_rows)

    anomaly_cols = [
        "HH1", "HH2", "LN", "province",
        "UB1D", "UB1M", "UB1Y",
        "IM6R1D", "IM6R1M", "IM6R1Y", "rv1_age_weeks",
        "IM6R2D", "IM6R2M", "IM6R2Y", "rv2_age_weeks",
        "rv1_before_birth", "rv1_under_2_weeks_flag", "rv1_over_52_weeks_flag",
        "rv2_before_birth", "rv2_under_2_weeks_flag", "rv2_over_52_weeks_flag",
        "dose2_before_dose1", "dose2_same_day_as_dose1",
    ]
    anomaly_mask = (
        qc["rv1_before_birth"] | qc["rv1_under_2_weeks_flag"] | qc["rv1_over_52_weeks_flag"] |
        qc["rv2_before_birth"] | qc["rv2_under_2_weeks_flag"] | qc["rv2_over_52_weeks_flag"] |
        qc["dose2_nonpositive_interval"]
    )
    qc.loc[anomaly_mask, anomaly_cols].to_csv(OUT / "mics_timing_anomaly_register.csv", index=False)

    print(json.dumps(national, indent=2))


if __name__ == "__main__":
    main()
