#!/usr/bin/env python3
"""Phase B1: Clark-aligned single imputation of Rotarix vaccination ages."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rotavirus_impact.config import SEED  # noqa: E402
from rotavirus_impact.mics_reconstruction import analytic_sample, reconstruct_rotavirus_coverage  # noqa: E402
from rotavirus_impact.sav_numeric import read_dictionary, read_numeric_subset  # noqa: E402
from rotavirus_impact.paths import raw_input_path  # noqa: E402
from rotavirus_impact.timing_imputation import (  # noqa: E402
    imputation_audit,
    imputation_diagnostics_by_cell,
    impute_vaccination_ages,
)

MICS_PATH = raw_input_path("mics")
OUT = ROOT / "outputs" / "timing"
OUT.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "HH1", "HH2", "LN", "CAGE", "CAGED", "HH6", "HH7", "CHWEIGHT", "WINDEX5",
    "UB1D", "UB1M", "UB1Y",
    "IM6R1D", "IM6R1M", "IM6R1Y", "IM6R2D", "IM6R2M", "IM6R2Y",
    "IM24", "IM25", "PSU", "STRATUM",
]


def weighted_stats(values: pd.Series, weights: pd.Series) -> dict[str, float | None]:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return {"weighted_mean_days": None}
    return {"weighted_mean_days": float((values[mask] * weights[mask]).sum() / weights[mask].sum())}


def main() -> None:
    _header, _vars, labels = read_dictionary(MICS_PATH)
    raw = read_numeric_subset(MICS_PATH, FIELDS)
    df = reconstruct_rotavirus_coverage(analytic_sample(raw))
    df["province"] = df["HH7"].map(labels["HH7"])
    out = impute_vaccination_ages(df, seed=SEED)

    audit = imputation_audit(out)
    audit["rv1"].update(weighted_stats(out.loc[out.rv1_vaccinated, "rv1_age_days"], out.loc[out.rv1_vaccinated, "CHWEIGHT"]))
    audit["rv2"].update(weighted_stats(out.loc[out.rv2_vaccinated, "rv2_age_days"], out.loc[out.rv2_vaccinated, "CHWEIGHT"]))
    (OUT / "imputation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    imputation_diagnostics_by_cell(out).to_csv(OUT / "imputation_diagnostics_by_cell.csv", index=False)

    rows = []
    for province, g in out.groupby("province", sort=True):
        row = {"province": province, "n_12_23_months": int(len(g))}
        for dose in (1, 2):
            vaccinated = g[f"rv{dose}_vaccinated"]
            ages = g.loc[vaccinated, f"rv{dose}_age_days"]
            row.update({
                f"rv{dose}_vaccinated_n": int(vaccinated.sum()),
                f"rv{dose}_observed_exact_n": int(g[f"rv{dose}_age_source"].eq("observed_exact_date").sum()),
                f"rv{dose}_imputed_n": int(g[f"rv{dose}_age_source"].eq("imputed").sum()),
                f"rv{dose}_median_age_weeks": float(ages.median() / 7.0) if len(ages) else None,
                f"rv{dose}_mean_age_weeks": float(ages.mean() / 7.0) if len(ages) else None,
                f"rv{dose}_p75_age_weeks": float(ages.quantile(0.75) / 7.0) if len(ages) else None,
            })
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "provincial_timing_postimputation.csv", index=False)

    # Aggregated evidence-source counts only; no row-level MICS microdata are exported.
    evidence_rows = []
    for dose in (1, 2):
        g = out.groupby(["province", f"rv{dose}_evidence", f"rv{dose}_age_source"], dropna=False).size().reset_index(name="n")
        g.insert(1, "dose", f"RV{dose}")
        g = g.rename(columns={f"rv{dose}_evidence": "evidence", f"rv{dose}_age_source": "age_source"})
        evidence_rows.append(g)
    pd.concat(evidence_rows, ignore_index=True).to_csv(OUT / "provincial_timing_evidence_counts.csv", index=False)

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
