#!/usr/bin/env python3
"""Phase E: Monte Carlo intervals and deterministic Paper-1 sensitivities."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rotavirus_impact.config import REFERENCE_RV_DEATHS_U5, SEED  # noqa: E402
from rotavirus_impact.impact_model import (  # noqa: E402
    INJECTABLE_NEXTGEN,
    ORAL_ROTARIX,
    RV1_TARGET_DAYS,
    RV2_TARGET_DAYS,
    impact_fraction,
    disease_age_mass,
    event_mass_from_conditional_shape,
    efficacy_kernel,
    load_curve_shapes,
    mean_timing_shape,
    step_timing_shape,
)
from rotavirus_impact.curve_parameter_uncertainty import (  # noqa: E402
    draw_params,
    timing_shape_from_params,
)
from rotavirus_impact.uncertainty_sensitivity import (  # noqa: E402
    MonteCarloSettings,
    effective_ve_scale,
    national_contrast_summary,
    national_summary_from_draws,
    provincial_summary,
    simulate_provincial_independent,
    simulate_shared_systemic_shocks,
)

CURVE_DIR = ROOT / "outputs" / "coverage_curves"
MORT_DIR = ROOT / "outputs" / "mortality"
IMPACT_DIR = ROOT / "outputs" / "impact"
OUT = ROOT / "outputs" / "uncertainty_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)


def _scenario_context():
    fits = pd.read_csv(CURVE_DIR / "coverage_curve_fits.csv")
    curves = pd.read_csv(CURVE_DIR / "coverage_curves_daily.csv")
    top8 = pd.read_csv(CURVE_DIR / "top8_rv1_26week_provinces.csv")["province"].tolist()
    mortality = pd.read_csv(MORT_DIR / "provincial_analytic_mortality_weights_33.csv")
    provinces = sorted(mortality.province.tolist())
    shapes = load_curve_shapes(curves, fits)
    top1 = mean_timing_shape(shapes, top8, "RV1")
    top2 = mean_timing_shape(shapes, top8, "RV2")
    step1 = step_timing_shape(RV1_TARGET_DAYS)
    step2 = step_timing_shape(RV2_TARGET_DAYS)
    f = fits[fits.province.isin(provinces)].pivot(index="province", columns="dose", values="final_crude_coverage")
    target1 = float(f.loc[top8, "RV1"].mean())
    target2 = float(f.loc[top8, "RV2"].mean())
    return fits, curves, top8, mortality, provinces, shapes, top1, top2, step1, step2, f, target1, target2


def _scenario_definitions(province, f, shapes, top1, top2, step1, step2, target1, target2):
    c1 = float(f.loc[province, "RV1"])
    c2 = float(f.loc[province, "RV2"])
    lifted1 = max(c1, target1)
    lifted2 = min(max(c2, target2), lifted1)
    own1, own2 = shapes[(province, "RV1")], shapes[(province, "RV2")]
    return {
        "S1": (c1, c2, own1, own2, ORAL_ROTARIX),
        "S2": (c1, c2, top1, top2, ORAL_ROTARIX),
        "S3": (lifted1, lifted2, own1, own2, ORAL_ROTARIX),
        "S4": (lifted1, lifted2, top1, top2, ORAL_ROTARIX),
        "S5": (1.0, 1.0, step1, step2, ORAL_ROTARIX),
        "S6": (c1, c2, own1, own2, INJECTABLE_NEXTGEN),
        "S7": (1.0, 1.0, step1, step2, INJECTABLE_NEXTGEN),
        "S8": (c1, c2, step1, step2, ORAL_ROTARIX),
    }


def deterministic_age_sensitivity():
    (_fits, _curves, _top8, mortality, provinces, shapes, top1, top2,
     step1, step2, f, target1, target2) = _scenario_context()
    baseline = mortality.set_index("province")["baseline_deaths_composite"]
    rows = []
    for age_dist in ("burr_primary", "hasso_very_high", "burr_deaths_shifted"):
        for province in provinces:
            for scenario, (c1, c2, s1, s2, product) in _scenario_definitions(
                province, f, shapes, top1, top2, step1, step2, target1, target2
            ).items():
                frac = impact_fraction(
                    rv1_final=c1, rv2_final=c2,
                    rv1_timing_shape=s1, rv2_timing_shape=s2,
                    product=product, age_distribution=age_dist,
                )
                rows.append({
                    "age_distribution": age_dist,
                    "province": province,
                    "scenario": scenario,
                    "impact_fraction": frac,
                    "deaths_averted": float(baseline.loc[province]) * frac,
                })
    prov = pd.DataFrame(rows)
    nat = prov.groupby(["age_distribution", "scenario"], as_index=False).agg(
        deaths_averted=("deaths_averted", "sum")
    )
    nat["baseline_deaths"] = float(baseline.sum())
    nat["impact_fraction"] = nat.deaths_averted / nat.baseline_deaths
    return prov, nat


def deterministic_mortality_weight_sensitivity():
    det = pd.read_csv(IMPACT_DIR / "provincial_primary_s1_s8.csv")
    mort = pd.read_csv(MORT_DIR / "provincial_analytic_mortality_weights_33.csv").set_index("province")
    schemes = {
        "uniform": "baseline_deaths_uniform",
        "composite_primary_linear_inverse": "baseline_deaths_composite",
        "composite_log_linear": "baseline_deaths_composite_log_linear",
        "composite_bounded_2x": "baseline_deaths_composite_bounded_2x",
        "composite_bounded_5x": "baseline_deaths_composite_bounded_5x",
        "ors_adjusted": "baseline_deaths_ors_adjusted",
        "care_seeking_adjusted": "baseline_deaths_care_seeking_adjusted",
        "underweight_adjusted": "baseline_deaths_underweight_adjusted",
    }
    rows = []
    for scheme, death_col in schemes.items():
        for row in det.itertuples(index=False):
            b = float(mort.loc[row.province, death_col])
            rows.append({
                "mortality_weighting": scheme,
                "province": row.province,
                "scenario": row.scenario,
                "baseline_deaths": b,
                "impact_fraction": float(row.impact_fraction),
                "deaths_averted": b * float(row.impact_fraction),
            })
    prov = pd.DataFrame(rows)
    nat = prov.groupby(["mortality_weighting", "scenario"], as_index=False).agg(
        baseline_deaths=("baseline_deaths", "sum"), deaths_averted=("deaths_averted", "sum")
    )
    nat["impact_fraction"] = nat.deaths_averted / nat.baseline_deaths
    return prov, nat


def urozgan_scope_sensitivity():
    """Compare 33-province renormalisation with keeping the 34-province allocation fixed.

    This does not invent Urozgan scenario results. It quantifies the effect of the
    analytical choice to redistribute the national 1,610-death burden across the
    remaining 33 provinces after excluding Urozgan from scenario analysis.
    """
    det = pd.read_csv(IMPACT_DIR / "provincial_primary_s1_s8.csv")
    full34 = pd.read_csv(MORT_DIR / "provincial_care_access_and_mortality_weights.csv").set_index("province")
    renorm33 = pd.read_csv(MORT_DIR / "provincial_analytic_mortality_weights_33.csv").set_index("province")
    rows = []
    for label, source in (("33_renormalized_to_1610", renorm33), ("34_allocation_then_drop_urozgan", full34)):
        for row in det.itertuples(index=False):
            b = float(source.loc[row.province, "baseline_deaths_composite"])
            rows.append({
                "analytic_scope": label,
                "province": row.province,
                "scenario": row.scenario,
                "baseline_deaths": b,
                "deaths_averted": b * float(row.impact_fraction),
            })
    p = pd.DataFrame(rows)
    n = p.groupby(["analytic_scope", "scenario"], as_index=False).agg(
        baseline_deaths=("baseline_deaths", "sum"), deaths_averted=("deaths_averted", "sum")
    )
    return p, n


def cfr_mapping_decomposition_sensitivity():
    """Cross alternative composite->CFR mappings with Urozgan scope choices.

    The decomposition ratio requested for robustness is the incremental coverage
    gain (S3-S1) divided by the top-eight timing-shape gain (S2-S1).  Because S8
    is an absolute on-time counterfactual rather than the realistic top-eight
    timing improvement, we also report coverage/(S8-S1) to avoid conflating the
    two timing contrasts.
    """
    det = pd.read_csv(IMPACT_DIR / "provincial_primary_s1_s8.csv")
    full34 = pd.read_csv(MORT_DIR / "provincial_care_access_and_mortality_weights.csv").set_index("province")
    renorm33 = pd.read_csv(MORT_DIR / "provincial_analytic_mortality_weights_33.csv").set_index("province")
    mapping_cols = {
        "linear_inverse": "baseline_deaths_composite",
        "log_linear": "baseline_deaths_composite_log_linear",
        "bounded_2x": "baseline_deaths_composite_bounded_2x",
        "bounded_5x": "baseline_deaths_composite_bounded_5x",
    }
    rows = []
    for mapping, col in mapping_cols.items():
        for scope, source in (
            ("33_renormalized_to_1610", renorm33),
            ("34_allocation_then_drop_urozgan", full34),
        ):
            x = det.copy()
            x["baseline_alt"] = x["province"].map(source[col])
            x["deaths_averted_alt"] = x["baseline_alt"] * x["impact_fraction"]
            nat = x.groupby("scenario")["deaths_averted_alt"].sum()
            base = x.loc[x.scenario.eq("S1"), "baseline_alt"].sum()
            timing_top8 = float(nat["S2"] - nat["S1"])
            coverage = float(nat["S3"] - nat["S1"])
            timing_exact = float(nat["S8"] - nat["S1"])
            multiplier_col = col.replace("baseline_deaths_", "cfr_multiplier_")
            rows.append({
                "cfr_mapping": mapping,
                "urozgan_scope": scope,
                "analytic_baseline_deaths": float(base),
                "timing_top8_gain_s2_minus_s1": timing_top8,
                "coverage_gain_s3_minus_s1": coverage,
                "timing_exact_gain_s8_minus_s1": timing_exact,
                "coverage_to_top8_timing_ratio": coverage / timing_top8,
                "coverage_to_exact_timing_ratio": coverage / timing_exact,
                "cfr_multiplier_min": float(source[multiplier_col].min()),
                "cfr_multiplier_max": float(source[multiplier_col].max()),
                "cfr_multiplier_max_min_ratio": float(source[multiplier_col].max() / source[multiplier_col].min()),
            })
    return pd.DataFrame(rows)


def monte_carlo_with_curve_parameter_uncertainty(independent: pd.DataFrame, settings: MonteCarloSettings):
    """Add conditional Clark curve-fit parameter uncertainty to primary MC.

    The stochastic specification is unchanged from v1.0.0: 100 independent
    whole-system curve-parameter realisations are generated and sampled with
    replacement across 500 VE/mortality replicates.  This implementation caches
    the dose-specific age-state protection components for each timing shape so
    that scenarios reusing the same shape do not repeat the same convolution.
    """
    fits, _curves, top8, mortality, provinces, _shapes, _top1, _top2, step1, step2, f, target1, target2 = _scenario_context()
    fit_idx = fits[fits.province.isin(provinces)].set_index(["province", "dose"])
    baseline_point = mortality.set_index("province")["baseline_deaths_composite"]
    rng = np.random.default_rng(settings.seed + 8080)
    curve_draw_count = 100

    disease = disease_age_mass("burr_primary")
    horizon = len(disease)
    oral_k1 = efficacy_kernel(ORAL_ROTARIX, ORAL_ROTARIX.one_dose_peak, horizon_days=horizon)
    oral_k2 = efficacy_kernel(ORAL_ROTARIX, ORAL_ROTARIX.two_dose_peak, horizon_days=horizon)

    def oral_components(shape, kernel):
        # F(a) is the fraction of eventual recipients vaccinated by age a;
        # avg(a) is their average protection at age a conditional on vaccination.
        event = event_mass_from_conditional_shape(shape, horizon_days=horizon)
        F = np.cumsum(event)
        conv = np.fft.irfft(
            np.fft.rfft(event, n=2*horizon-1) * np.fft.rfft(kernel, n=2*horizon-1),
            n=2*horizon-1,
        )[:horizon]
        with np.errstate(divide="ignore", invalid="ignore"):
            avg = np.where(F > 0, conv / F, 0.0)
        return F, avg

    def oral_fraction(c1, c2, comp1, comp2):
        F1, avg1 = comp1; F2, avg2 = comp2
        dose1_only = np.maximum(c1 * F1 - c2 * F2, 0.0)
        protection = dose1_only * avg1 + (c2 * F2) * avg2
        return float(np.dot(disease, np.clip(protection, 0.0, 1.0)))

    def parenteral_fraction(c1, c2, shape1, shape2):
        # Use the same mutually exclusive dose-state mechanics as the oral
        # product. With identical 94% non-waning one- and two-dose profiles,
        # protection starts after RV1 and is unchanged by the RV2 transition.
        e1 = event_mass_from_conditional_shape(shape1, horizon_days=horizon)
        e2 = event_mass_from_conditional_shape(shape2, horizon_days=horizon)
        F1 = np.cumsum(e1); F2 = np.cumsum(e2)
        dose1_only = np.maximum(c1 * F1 - c2 * F2, 0.0)
        protection = (INJECTABLE_NEXTGEN.one_dose_peak * dose1_only
                      + INJECTABLE_NEXTGEN.two_dose_peak * c2 * F2)
        return float(np.dot(disease, np.clip(protection, 0.0, 1.0)))

    step1_comp = oral_components(step1, oral_k1)
    step2_comp = oral_components(step2, oral_k2)
    s5_frac = oral_fraction(1.0, 1.0, step1_comp, step2_comp)
    s7_frac = parenteral_fraction(1.0, 1.0, step1, step2)

    lib_rows = []
    diag = {"parameter_draws": 0, "clipped_fallbacks": 0, "total_rejection_attempts": 0, "curve_realisations": curve_draw_count}
    for draw_id in range(curve_draw_count):
        sampled_shapes = {}
        for province in provinces:
            for dose, target in (("RV1", RV1_TARGET_DAYS), ("RV2", RV2_TARGET_DAYS)):
                params, info = draw_params(fit_idx.loc[(province, dose)], rng)
                sampled_shapes[(province, dose)] = timing_shape_from_params(params, target)
                diag["parameter_draws"] += 1
                diag["clipped_fallbacks"] += int(info["clipped_fallback"])
                diag["total_rejection_attempts"] += int(info["attempts"] - 1)
        top1 = np.mean([sampled_shapes[(p, "RV1")] for p in top8], axis=0)
        top2 = np.mean([sampled_shapes[(p, "RV2")] for p in top8], axis=0)
        top1_comp = oral_components(top1, oral_k1)
        top2_comp = oral_components(top2, oral_k2)

        own_comp = {}
        for province in provinces:
            own_comp[(province, "RV1")] = oral_components(sampled_shapes[(province, "RV1")], oral_k1)
            own_comp[(province, "RV2")] = oral_components(sampled_shapes[(province, "RV2")], oral_k2)

        for province in provinces:
            c1 = float(f.loc[province, "RV1"]); c2 = float(f.loc[province, "RV2"])
            lifted1 = max(c1, target1); lifted2 = min(max(c2, target2), lifted1)
            oc1, oc2 = own_comp[(province, "RV1")], own_comp[(province, "RV2")]
            own1, own2 = sampled_shapes[(province, "RV1")], sampled_shapes[(province, "RV2")]
            fracs = {
                "S1": (oral_fraction(c1, c2, oc1, oc2), ORAL_ROTARIX.name),
                "S2": (oral_fraction(c1, c2, top1_comp, top2_comp), ORAL_ROTARIX.name),
                "S3": (oral_fraction(lifted1, lifted2, oc1, oc2), ORAL_ROTARIX.name),
                "S4": (oral_fraction(lifted1, lifted2, top1_comp, top2_comp), ORAL_ROTARIX.name),
                "S5": (s5_frac, ORAL_ROTARIX.name),
                "S6": (parenteral_fraction(c1, c2, own1, own2), INJECTABLE_NEXTGEN.name),
                "S7": (s7_frac, INJECTABLE_NEXTGEN.name),
                "S8": (oral_fraction(c1, c2, step1_comp, step2_comp), ORAL_ROTARIX.name),
            }
            for scenario, (frac_curve, product_name) in fracs.items():
                lib_rows.append({
                    "curve_draw": draw_id, "province": province, "scenario": scenario,
                    "product": product_name, "curve_impact_fraction": frac_curve,
                })
    library = pd.DataFrame(lib_rows)
    library.to_csv(OUT / "curve_fraction_library.csv", index=False)

    selected = pd.DataFrame({
        "replicate": np.arange(settings.replicates, dtype=int),
        "curve_draw": rng.integers(0, curve_draw_count, size=settings.replicates),
    })
    selected.to_csv(OUT / "curve_draw_selection.csv", index=False)
    x = selected.merge(library, on="curve_draw", how="left", validate="many_to_many")
    shock = independent[["province", "replicate", "mortality_scale", "ve_scale_raw"]].drop_duplicates()
    x = x.merge(shock, on=["province", "replicate"], how="left", validate="many_to_one")
    x["baseline_point"] = x["province"].map(baseline_point)
    x["baseline_deaths"] = x["baseline_point"] * x["mortality_scale"]

    oral_mask = x["product"].eq("oral_rotarix")
    max_oral = 1.0 / ORAL_ROTARIX.two_dose_peak
    max_inj = 1.0 / INJECTABLE_NEXTGEN.two_dose_peak
    x["ve_scale_effective"] = np.where(
        oral_mask,
        np.minimum(np.maximum(x["ve_scale_raw"], settings.ve_scale_floor), max_oral),
        np.minimum(np.maximum(x["ve_scale_raw"], settings.ve_scale_floor), max_inj),
    )
    x["impact_fraction"] = np.clip(x["curve_impact_fraction"] * x["ve_scale_effective"], 0.0, 1.0)
    x["deaths_averted"] = x["baseline_deaths"] * x["impact_fraction"]
    x["residual_deaths"] = x["baseline_deaths"] - x["deaths_averted"]
    return x[[
        "province", "replicate", "scenario", "product", "baseline_deaths",
        "impact_fraction", "deaths_averted", "residual_deaths"
    ]], diag


def main() -> None:
    deterministic = pd.read_csv(IMPACT_DIR / "provincial_primary_s1_s8.csv")
    settings = MonteCarloSettings(replicates=500, seed=SEED)

    independent = simulate_provincial_independent(deterministic, settings)
    prov_ci = provincial_summary(independent, deterministic)
    nat_ci = national_summary_from_draws(independent, deterministic, "province_independent_primary")
    nat_contrasts = national_contrast_summary(independent, deterministic, "province_independent_ve_mortality_only")
    nat_ci["uncertainty_structure"] = "province_independent_ve_mortality_only"

    combined, curve_diag = monte_carlo_with_curve_parameter_uncertainty(independent, settings)
    combined_nat_wide = combined.groupby(["replicate", "scenario"])["deaths_averted"].sum().unstack("scenario")
    combined_nat_wide.reset_index().to_csv(OUT / "primary_national_scenario_draws.csv", index=False)
    ratio_draws = pd.DataFrame({
        "replicate": combined_nat_wide.index.to_numpy(),
        "coverage_to_top8_timing_ratio": (combined_nat_wide["S3"] - combined_nat_wide["S1"]) / (combined_nat_wide["S2"] - combined_nat_wide["S1"]),
        "coverage_to_exact_timing_ratio": (combined_nat_wide["S3"] - combined_nat_wide["S1"]) / (combined_nat_wide["S8"] - combined_nat_wide["S1"]),
    })
    ratio_draws.to_csv(OUT / "decomposition_ratio_mc_draws.csv", index=False)
    ratio_summary = pd.DataFrame([
        {
            "ratio": col,
            "median": float(ratio_draws[col].median()),
            "lower_95": float(ratio_draws[col].quantile(0.025)),
            "upper_95": float(ratio_draws[col].quantile(0.975)),
            "min": float(ratio_draws[col].min()),
            "max": float(ratio_draws[col].max()),
        }
        for col in ("coverage_to_top8_timing_ratio", "coverage_to_exact_timing_ratio")
    ])
    ratio_summary.to_csv(OUT / "decomposition_ratio_mc_summary.csv", index=False)
    prov_ci_curve = provincial_summary(combined, deterministic)
    nat_ci_curve = national_summary_from_draws(
        combined, deterministic, "province_independent_plus_curve_fit_parameters"
    )
    nat_contrasts_curve = national_contrast_summary(
        combined, deterministic, "province_independent_plus_curve_fit_parameters"
    )

    shared = simulate_shared_systemic_shocks(deterministic, settings)
    nat_shared = national_summary_from_draws(shared, deterministic, "fully_shared_systemic_sensitivity")
    contrast_shared = national_contrast_summary(shared, deterministic, "fully_shared_systemic_sensitivity")

    prov_ci.to_csv(OUT / "provincial_ve_mortality_only_95_intervals.csv", index=False)
    prov_ci_curve.to_csv(OUT / "provincial_primary_95_intervals.csv", index=False)
    pd.concat([nat_ci, nat_ci_curve, nat_shared], ignore_index=True).to_csv(OUT / "national_95_intervals.csv", index=False)
    pd.concat([nat_contrasts, nat_contrasts_curve, contrast_shared], ignore_index=True).to_csv(OUT / "national_contrast_95_intervals.csv", index=False)

    age_prov, age_nat = deterministic_age_sensitivity()
    age_prov.to_csv(OUT / "provincial_age_distribution_sensitivity.csv", index=False)
    age_nat.to_csv(OUT / "national_age_distribution_sensitivity.csv", index=False)
    age_rows = []
    for age_dist, g in age_nat.groupby("age_distribution"):
        z = g.set_index("scenario")["deaths_averted"]
        timing = float(z["S2"] - z["S1"])
        coverage = float(z["S3"] - z["S1"])
        age_rows.append({
            "age_distribution": age_dist,
            "s1_deaths_averted": float(z["S1"]),
            "timing_gain_s2_minus_s1": timing,
            "coverage_gain_s3_minus_s1": coverage,
            "coverage_to_timing_ratio": coverage / timing,
            "exact_timing_gain_s8_minus_s1": float(z["S8"] - z["S1"]),
            "parenteral_gain_s6_minus_s1": float(z["S6"] - z["S1"]),
        })
    pd.DataFrame(age_rows).to_csv(OUT / "age_distribution_contrast_sensitivity.csv", index=False)

    mort_prov, mort_nat = deterministic_mortality_weight_sensitivity()
    mort_prov.to_csv(OUT / "provincial_mortality_weight_sensitivity.csv", index=False)
    mort_nat.to_csv(OUT / "national_mortality_weight_sensitivity.csv", index=False)

    scope_prov, scope_nat = urozgan_scope_sensitivity()
    scope_prov.to_csv(OUT / "provincial_urozgan_scope_sensitivity.csv", index=False)
    scope_nat.to_csv(OUT / "national_urozgan_scope_sensitivity.csv", index=False)

    cfr_map = cfr_mapping_decomposition_sensitivity()
    cfr_map.to_csv(OUT / "cfr_mapping_decomposition_sensitivity.csv", index=False)

    # Audit key invariants and quantify implications of assumptions.
    ve_raw = independent[["province", "replicate", "ve_scale_raw"]].drop_duplicates()
    audit = {
        "mc_replicates": settings.replicates,
        "seed": settings.seed,
        "primary_national_interval_structure": "province-specific independent mortality and VE scale draws; scenario draws correlated within province",
        "shared_systemic_sensitivity": "same mortality and VE scale draw applied to every province within a replicate",
        "intervals_exclude": [
            "MICS sampling uncertainty",
            "single-imputation uncertainty",
            "disease-age parameter uncertainty",
            "mortality-weighting structural uncertainty",
        ],
        "decomposition_ratio_mc_summary": ratio_summary.to_dict(orient="records"),
        "coverage_curve_parameter_uncertainty": {
            "included_in_primary_intervals": True,
            "method": "multivariate normal draws from conditional nonlinear least-squares covariance of Clark five-parameter fit; draws constrained to parameter bounds",
            "important_limitation": "does not equal a full design-based MICS sampling variance and does not propagate final crude coverage or imputation uncertainty",
            **curve_diag,
        },
        "ve_draw_floor_count": int((ve_raw.ve_scale_raw <= settings.ve_scale_floor + 1e-15).sum()),
        "injectable_ve_cap_threshold_scale": float(1.0 / 0.94),
        "independent_draw_rows": int(len(independent)),
        "national_primary_baseline_point": float(deterministic[deterministic.scenario.eq("S1")].baseline_deaths.sum()),
        "urozgan_baseline_deaths_in_full34_composite": float(
            pd.read_csv(MORT_DIR / "provincial_care_access_and_mortality_weights.csv")
            .loc[lambda x: x.province.str.upper().eq("UROZGAN"), "baseline_deaths_composite"].iloc[0]
        ),
        "full34_then_drop_baseline_sum": float(
            scope_nat.loc[(scope_nat.analytic_scope == "34_allocation_then_drop_urozgan") & (scope_nat.scenario == "S1"), "baseline_deaths"].iloc[0]
        ),
    }
    (OUT / "uncertainty_sensitivity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("Primary national intervals including curve-fit parameter uncertainty")
    print(nat_ci_curve.to_string(index=False))
    print("\nVE + mortality only intervals")
    print(nat_ci.to_string(index=False))
    print("\nPrimary contrast intervals")
    print(nat_contrasts.to_string(index=False))
    print("\nAge distribution sensitivity")
    print(age_nat.to_string(index=False))
    print("\nMortality weighting sensitivity")
    print(mort_nat.to_string(index=False))
    print("\nUrozgan scope sensitivity")
    print(scope_nat.to_string(index=False))
    print("\nCFR mapping decomposition sensitivity")
    print(cfr_map.to_string(index=False))
    print("\nAudit")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
