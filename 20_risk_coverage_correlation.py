#!/usr/bin/env python3
"""Descriptive association between model-assigned mortality risk and vaccine coverage."""
from pathlib import Path
import sys
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/'reporting'; OUT.mkdir(parents=True,exist_ok=True)
fits=pd.read_csv(ROOT/'outputs'/'coverage_curves'/'coverage_curve_fits.csv')
mort=pd.read_csv(ROOT/'outputs'/'mortality'/'provincial_analytic_mortality_weights_33.csv')
f=fits[fits.province.isin(mort.province)].pivot(index='province',columns='dose',values='final_crude_coverage')
x=mort.set_index('province').join(f[['RV1','RV2']])
rows=[]
for dose in ['RV1','RV2']:
    for variable in ['cfr_multiplier_composite','care_access_composite_z']:
        rho,p=spearmanr(x[variable],x[dose])
        r,pp=pearsonr(x[variable],x[dose])
        rows.append({'risk_variable':variable,'coverage_measure':dose,'n_provinces':len(x),'spearman_rho':rho,'spearman_p':p,'pearson_r':r,'pearson_p':pp})
pd.DataFrame(rows).to_csv(OUT/'risk_coverage_correlation.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
