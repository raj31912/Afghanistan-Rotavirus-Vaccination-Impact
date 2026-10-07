#!/usr/bin/env python3
"""Independent face-validity audit of the MICS-derived provincial CFR ranking.

No external reference indicator in this module contributes to primary mortality
weights. The AHS 2018 service-coverage indicators are used only to ask whether
provinces with poor independent health-service coverage also tend to receive
higher MICS-derived relative CFR multipliers.
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/'face_validity'; OUT.mkdir(parents=True,exist_ok=True)
MORT=ROOT/'outputs'/'mortality'/'provincial_care_access_and_mortality_weights.csv'
AHS=ROOT/'data'/'external_reference'/'ahs2018_table7_health_service_coverage.csv'

EXPECTED_HIGH={'HELMAND','BADGHIS','GHOR'}
EXPECTED_LOW={'BAMYAN','DAYKUNDI','KABUL'}


def z(x):
    x=x.astype(float)
    return (x-x.mean())/x.std(ddof=0)


def main():
    mort=pd.read_csv(MORT)
    ahs=pd.read_csv(AHS)
    service_cols=['modern_contraceptive','penta3','any_anc','c_section','institutional_delivery','any_pnc','tetanus2plus']
    for c in service_cols:
        ahs['z_'+c]=z(ahs[c])
    ahs['ahs_service_coverage_z']=ahs[['z_'+c for c in service_cols]].mean(axis=1)
    ahs['ahs_poor_access_risk_z']=-ahs['ahs_service_coverage_z']
    x=mort.merge(ahs[['province','ahs_service_coverage_z','ahs_poor_access_risk_z']+service_cols],on='province',how='inner',validate='one_to_one')
    x['mics_cfr_rank_high_to_low']=x['cfr_multiplier_composite'].rank(ascending=False,method='min').astype(int)
    x['ahs_poor_access_rank_high_to_low']=x['ahs_poor_access_risk_z'].rank(ascending=False,method='min').astype(int)
    rho,p=spearmanr(x['cfr_multiplier_composite'],x['ahs_poor_access_risk_z'])
    x['rank_difference_mics_minus_ahs']=x['mics_cfr_rank_high_to_low']-x['ahs_poor_access_rank_high_to_low']
    x.sort_values('mics_cfr_rank_high_to_low').to_csv(OUT/'cfr_face_validity_vs_ahs2018.csv',index=False)

    primary_high=x.nsmallest(3,'mics_cfr_rank_high_to_low')['province'].tolist()
    primary_low=x.nlargest(3,'mics_cfr_rank_high_to_low')['province'].tolist()
    ahs_high=x.nsmallest(3,'ahs_poor_access_rank_high_to_low')['province'].tolist()
    ahs_low=x.nlargest(3,'ahs_poor_access_rank_high_to_low')['province'].tolist()
    focal=x[x.province.isin(EXPECTED_HIGH|EXPECTED_LOW)].copy().sort_values('mics_cfr_rank_high_to_low')
    focal.to_csv(OUT/'andy_six_province_face_validity.csv',index=False)
    summary={
        'primary_mapping':'linear_inverse',
        'important_note':'All monotone composite-to-CFR mappings preserve the same province ordering; this audit evaluates the composite ranking, not the mapping spread.',
        'spearman_mics_cfr_vs_ahs_poor_access_risk':float(rho),
        'spearman_p_value':float(p),
        'mics_top3_cfr':primary_high,
        'mics_bottom3_cfr':primary_low,
        'ahs2018_top3_poor_service_coverage':ahs_high,
        'ahs2018_bottom3_poor_service_coverage':ahs_low,
        'andy_expected_high_cfr':sorted(EXPECTED_HIGH),
        'andy_expected_low_cfr':sorted(EXPECTED_LOW),
        'mics_top3_overlap_with_andy_high':len(set(primary_high)&EXPECTED_HIGH),
        'mics_bottom3_overlap_with_andy_low':len(set(primary_low)&EXPECTED_LOW),
        'ahs_top3_overlap_with_andy_high':len(set(ahs_high)&EXPECTED_HIGH),
        'ahs_bottom3_overlap_with_andy_low':len(set(ahs_low)&EXPECTED_LOW),
        'interpretation':'External AHS service coverage supports Helmand/Ghor/Badghis as poor-access provinces much more strongly than the three-variable MICS composite ranks them. This discrepancy is retained as a structural limitation; no weights are tuned to force the expected ranking.'
    }
    (OUT/'face_validity_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('\nFocal provinces:\n',focal[['province','cfr_multiplier_composite','mics_cfr_rank_high_to_low','ahs_poor_access_rank_high_to_low','ors_percent','formal_care_percent','underweight_percent']].to_string(index=False))

if __name__=='__main__': main()
