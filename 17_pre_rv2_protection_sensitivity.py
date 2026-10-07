#!/usr/bin/env python3
"""Structural sensitivity withholding oral protection before RV2 for eventual two-dose recipients."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from rotavirus_impact.impact_model import (
    ORAL_ROTARIX, INJECTABLE_NEXTGEN, RV1_TARGET_DAYS, RV2_TARGET_DAYS,
    load_curve_shapes, mean_timing_shape, step_timing_shape, impact_fraction,
    disease_age_mass, event_mass_from_conditional_shape, efficacy_kernel,
)
CUR=ROOT/'outputs'/'coverage_curves'; MORT=ROOT/'outputs'/'mortality'; OUT=ROOT/'outputs'/'uncertainty_sensitivity'


def oral_no_pre_fraction(c1,c2,s1,s2):
    disease=disease_age_mass('burr_primary'); horizon=len(disease)
    e1=event_mass_from_conditional_shape(s1,horizon_days=horizon)
    e2=event_mass_from_conditional_shape(s2,horizon_days=horizon)
    k1=efficacy_kernel(ORAL_ROTARIX,ORAL_ROTARIX.one_dose_peak,horizon_days=horizon)
    k2=efficacy_kernel(ORAL_ROTARIX,ORAL_ROTARIX.two_dose_peak,horizon_days=horizon)
    one_only=max(float(c1)-float(c2),0.0)
    p1=fftconvolve(e1,k1,mode='full')[:horizon]*one_only
    p2=fftconvolve(e2,k2,mode='full')[:horizon]*float(c2)
    return float(np.dot(disease,np.clip(p1+p2,0,1)))


def main():
    fits=pd.read_csv(CUR/'coverage_curve_fits.csv'); curves=pd.read_csv(CUR/'coverage_curves_daily.csv')
    shapes=load_curve_shapes(curves,fits); mort=pd.read_csv(MORT/'provincial_analytic_mortality_weights_33.csv').set_index('province')
    provinces=sorted(mort.index); f=fits[fits.province.isin(provinces)].pivot(index='province',columns='dose',values='final_crude_coverage')
    top8=pd.read_csv(CUR/'top8_rv1_26week_provinces.csv').province.tolist(); top1=mean_timing_shape(shapes,top8,'RV1'); top2=mean_timing_shape(shapes,top8,'RV2')
    step1=step_timing_shape(RV1_TARGET_DAYS); step2=step_timing_shape(RV2_TARGET_DAYS); target1=float(f.loc[top8,'RV1'].mean()); target2=float(f.loc[top8,'RV2'].mean())
    rows=[]
    for p in provinces:
        c1=float(f.loc[p,'RV1']); c2=float(f.loc[p,'RV2']); l1=max(c1,target1); l2=min(max(c2,target2),l1); o1=shapes[(p,'RV1')]; o2=shapes[(p,'RV2')]; base=float(mort.loc[p,'baseline_deaths_composite'])
        oral_defs={'S1':(c1,c2,o1,o2),'S2':(c1,c2,top1,top2),'S3':(l1,l2,o1,o2),'S4':(l1,l2,top1,top2),'S5':(1,1,step1,step2),'S8':(c1,c2,step1,step2)}
        for s,(a,b,t1,t2) in oral_defs.items():
            rows.append({'province':p,'scenario':s,'deaths_averted':base*oral_no_pre_fraction(a,b,t1,t2)})
        for s,(a,b,t1,t2) in {'S6':(c1,c2,o1,o2),'S7':(1,1,step1,step2)}.items():
            frac=impact_fraction(rv1_final=a,rv2_final=b,rv1_timing_shape=t1,rv2_timing_shape=t2,product=INJECTABLE_NEXTGEN,age_distribution='burr_primary')
            rows.append({'province':p,'scenario':s,'deaths_averted':base*frac})
    x=pd.DataFrame(rows); nat=x.groupby('scenario').deaths_averted.sum(); primary=pd.read_csv(ROOT/'outputs/impact/national_primary_s1_s8.csv').set_index('scenario').deaths_averted
    order=[f'S{i}' for i in range(1,9)]
    out=pd.DataFrame({'scenario':order,'no_pre_rv2_sensitivity':[nat[s] for s in order],'sequential_primary':[primary[s] for s in order]})
    out['difference_sequential_minus_no_pre']=out.sequential_primary-out.no_pre_rv2_sensitivity
    out.to_csv(OUT/'pre_rv2_protection_sensitivity.csv',index=False)
    print(out.to_string(index=False))

if __name__=='__main__': main()
