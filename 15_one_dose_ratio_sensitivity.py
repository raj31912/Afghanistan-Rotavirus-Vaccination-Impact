#!/usr/bin/env python3
from pathlib import Path
import sys
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from rotavirus_impact.impact_model import ProductProfile, INJECTABLE_NEXTGEN, RV1_TARGET_DAYS,RV2_TARGET_DAYS,load_curve_shapes,mean_timing_shape,step_timing_shape,impact_fraction
CUR=ROOT/'outputs'/'coverage_curves'; MORT=ROOT/'outputs'/'mortality'; OUT=ROOT/'outputs'/'uncertainty_sensitivity'
fits=pd.read_csv(CUR/'coverage_curve_fits.csv'); curves=pd.read_csv(CUR/'coverage_curves_daily.csv')
mort=pd.read_csv(MORT/'provincial_analytic_mortality_weights_33.csv').set_index('province')
provinces=sorted(mort.index); shapes=load_curve_shapes(curves,fits); f=fits[fits.province.isin(provinces)].pivot(index='province',columns='dose',values='final_crude_coverage')
top8=pd.read_csv(CUR/'top8_rv1_26week_provinces.csv').province.tolist(); top1=mean_timing_shape(shapes,top8,'RV1'); top2=mean_timing_shape(shapes,top8,'RV2'); step1=step_timing_shape(RV1_TARGET_DAYS); step2=step_timing_shape(RV2_TARGET_DAYS)
target1=float(f.loc[top8,'RV1'].mean()); target2=float(f.loc[top8,'RV2'].mean())
rows=[]
for ratio in [0.55,0.75,0.96,1.0]:
  oral=ProductProfile(name='oral_rotarix',one_dose_peak=ratio,two_dose_peak=1.0,waning=True)
  for p in provinces:
    c1=float(f.loc[p,'RV1']); c2=float(f.loc[p,'RV2']); l1=max(c1,target1); l2=min(max(c2,target2),l1); own1=shapes[(p,'RV1')]; own2=shapes[(p,'RV2')]
    defs={'S1':(c1,c2,own1,own2,oral),'S2':(c1,c2,top1,top2,oral),'S3':(l1,l2,own1,own2,oral),'S4':(l1,l2,top1,top2,oral),'S5':(1,1,step1,step2,oral),'S6':(c1,c2,own1,own2,INJECTABLE_NEXTGEN),'S7':(1,1,step1,step2,INJECTABLE_NEXTGEN),'S8':(c1,c2,step1,step2,oral)}
    b=float(mort.loc[p,'baseline_deaths_composite'])
    for s,(a,b2,t1,t2,prod) in defs.items():
      frac=impact_fraction(rv1_final=a,rv2_final=b2,rv1_timing_shape=t1,rv2_timing_shape=t2,product=prod,age_distribution='burr_primary')
      rows.append({'one_dose_to_two_dose_ratio':ratio,'province':p,'scenario':s,'impact_fraction':frac,'deaths_averted':b*frac})
x=pd.DataFrame(rows); x.to_csv(OUT/'one_dose_ratio_provincial_sensitivity.csv',index=False)
n=x.groupby(['one_dose_to_two_dose_ratio','scenario'],as_index=False).deaths_averted.sum(); n.to_csv(OUT/'one_dose_ratio_national_sensitivity.csv',index=False)
wide=n.pivot(index='one_dose_to_two_dose_ratio',columns='scenario',values='deaths_averted')
summary=pd.DataFrame({'one_dose_to_two_dose_ratio':wide.index,'S1':wide.S1,'S2_minus_S1':wide.S2-wide.S1,'S3_minus_S1':wide.S3-wide.S1,'S4_minus_S1':wide.S4-wide.S1,'S8_minus_S1':wide.S8-wide.S1,'S6_minus_S1':wide.S6-wide.S1,'coverage_to_top8_timing_ratio':(wide.S3-wide.S1)/(wide.S2-wide.S1),'coverage_to_exact_timing_ratio':(wide.S3-wide.S1)/(wide.S8-wide.S1)})
summary.to_csv(OUT/'one_dose_ratio_contrast_sensitivity.csv',index=False)
print(summary.to_string(index=False))
