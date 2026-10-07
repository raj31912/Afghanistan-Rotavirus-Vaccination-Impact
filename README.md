# Afghanistan rotavirus vaccination impact analysis

The analysis supporting the provincial rotavirus vaccination impact study in Afghanistan.

The pipeline reconstructs RV1/RV2 coverage and timing from Afghanistan MICS 2022–23, fits age-specific uptake curves, allocates a fixed national mortality reference across provinces, evaluates delivery and product counterfactuals, propagates uncertainty, and runs structural sensitivity analyses.

## Main analytical choices

- The primary analysis includes 33 provinces; Urozgan is retained descriptively but excluded from timing-based scenarios because dated vaccination evidence is sparse.
- Oral-vaccine protection follows the Afghanistan-specific Anwari parameterisation: 100% initial modelled protection after each dose, gamma waning with mean 10 months and shape 3, and sequential protection from RV1 until transition to RV2.
- The hypothetical parenteral counterfactual uses 94% non-waning protection with the same sequential dose-state mechanics, so product comparisons do not change the age at which protection begins.
- The primary mortality allocation uses under-five population and a MICS-derived care/vulnerability composite. Alternative mappings are structural sensitivities, not observed provincial mortality estimates.
- The national-average comparison is restricted to the same 33 analytic provinces and uses under-five-population-weighted observed delivery, making the uniform-mortality comparison identical by construction.
- Primary uncertainty uses 500 Monte Carlo replicates (seed 42) with PERT-Beta inputs for mortality and initial oral-vaccine protection plus conditional timing-curve parameter uncertainty.

## Repository layout

The numbered scripts reproduce the analysis in sequence. `src/rotavirus_impact/` contains reusable model functions and `tests/` contains validation tests. Generated results are written to `outputs/`; publication figures are written to `figures/`.

Additional sensitivity stages include:

- `15_one_dose_ratio_sensitivity.py` — one-dose:two-dose effectiveness ratios.
- `16_source_aligned_uncertainty.py` — primary Monte Carlo uncertainty analysis.
- `17_pre_rv2_protection_sensitivity.py` — sensitivity to withholding protection before RV2.
- `18_benchmark_size_sensitivity.py` — top-5/top-8/top-10 empirical benchmark choice.
- `19_same_scope_national_average.py` — independent check of the 33-province aggregation comparison.
- `20_risk_coverage_correlation.py` — descriptive association between model-assigned mortality risk and provincial coverage.

## Requirements

Python 3.11 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Source data

Restricted source data are not distributed in the public repository. A complete source-to-results reproduction requires:

1. Afghanistan MICS 2022–23 child SPSS file.
2. Afghanistan ADM1 under-five population file.
3. The Afghanistan UNIVAC workbook used for provenance and compatibility checks.

The UNIVAC workbook is not treated as the empirical source for vaccine effectiveness, waning, disease-age, or mortality parameters; those parameters are tied to the published sources documented in `docs/PARAMETER_PROVENANCE.md`.

Place source files in `data/raw/` using the filenames documented in `data/raw/README.md`, or pass paths to `run_all.py`.

## Run the complete analysis

```bash
python run_all.py \
  --mics /path/to/ch2022.sav \
  --population /path/to/afg_admpop_adm1_2026.csv \
  --univac /path/to/UNIVAC_Afghanistan.xlsb
```

Run the validation suite with:

```bash
pytest -q
```

Tests that require restricted MICS or UNIVAC source files skip automatically when those files are unavailable.

## Public release

The public deposit should contain code, tests, documentation, aggregate/non-identifying results, and figure source data only. Raw MICS records, UNIVAC files, and record-linked quality-control outputs must not be distributed.

The permanent repository identifier is assigned by the repository at deposit and should be cited in the manuscript once available.

## License and citation

Original analysis software and documentation in this public release are distributed under the MIT License; see `LICENSE`. Restricted or third-party source materials, including Afghanistan MICS microdata and UNIVAC files, are not relicensed or distributed here.

Citation metadata are provided in `CITATION.cff`. Add the permanent repository identifier/DOI to the repository record and manuscript once assigned.
