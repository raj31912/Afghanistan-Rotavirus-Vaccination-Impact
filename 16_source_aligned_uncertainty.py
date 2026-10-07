#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from rotavirus_impact.config import SEED
from rotavirus_impact.impact_model import ORAL_ROTARIX,RV1_TARGET_DAYS,RV2_TARGET_DAYS,impact_fraction,load_curve_shapes
from rotavirus_impact.curve_parameter_uncertainty import draw_params,timing_shape_from_params

UNC=ROOT/'outputs'/'uncertainty_sensitivity'; CUR=ROOT/'outputs'/'coverage_curves'; IMP=ROOT/'outputs'/'impact'; MORT=ROOT/'outputs'/'mortality'; OUT=ROOT/'outputs'/'reporting'; OUT.mkdir(exist_ok=True)
N=500

def pert(rng, lo, mode, hi, n, lam=4.0):
    if hi <= lo: return np.full(n, mode, dtype=float)
    a=1+lam*(mode-lo)/(hi-lo); b=1+lam*(hi-mode)/(hi-lo)
    return lo+(hi-lo)*rng.beta(a,b,size=n)

# Source-based uncertainty inputs from Anwari et al. 2025:
# RVGE mortality rate 26 (22-30) per 100,000 <5; initial VE 100% (45-100%).
rng=np.random.default_rng(SEED+1616)
mort_rate=pert(rng,22.0,26.0,30.0,N); mort_scale=mort_rate/26.0
oral_initial_ve=pert(rng,0.45,1.0,1.0,N)

lib=pd.read_csv(UNC/'curve_fraction_library.csv'); selected=pd.read_csv(UNC/'curve_draw_selection.csv')
x=selected.merge(lib,on='curve_draw',how='left',validate='many_to_many')
det=pd.read_csv(IMP/'provincial_primary_s1_s8.csv')
base=det[det.scenario.eq('S1')].set_index('province')['baseline_deaths']
x['baseline_point']=x.province.map(base)
x['mort_scale']=x.replicate.map(pd.Series(mort_scale,index=np.arange(N)))
x['oral_ve_scale']=x.replicate.map(pd.Series(oral_initial_ve,index=np.arange(N)))
x['baseline_deaths']=x.baseline_point*x.mort_scale
x['bio_scale']=np.where(x['product'].eq('oral_rotarix'),x.oral_ve_scale,1.0)
x['impact_fraction']=np.clip(x.curve_impact_fraction*x.bio_scale,0,1)
x['deaths_averted']=x.baseline_deaths*x.impact_fraction
x['residual_deaths']=x.baseline_deaths-x.deaths_averted

# Province-specific source-aligned intervals for supplementary reporting.
prov_points=det.set_index(['province','scenario'])['deaths_averted']
prov_rows=[]
for (province,scenario),g in x.groupby(['province','scenario']):
    a=g['deaths_averted'].to_numpy(float)
    prov_rows.append({
        'province':province,
        'scenario':scenario,
        'point_deaths_averted':float(prov_points.loc[(province,scenario)]),
        'mc_mean':float(a.mean()),
        'mc_median':float(np.median(a)),
        'lower_95':float(np.quantile(a,.025)),
        'upper_95':float(np.quantile(a,.975)),
    })
pd.DataFrame(prov_rows).sort_values(['province','scenario']).to_csv(UNC/'source_aligned_provincial_95_intervals.csv',index=False)

nat=x.groupby(['replicate','scenario'],as_index=False).deaths_averted.sum(); wide=nat.pivot(index='replicate',columns='scenario',values='deaths_averted')
wide.reset_index().to_csv(UNC/'source_aligned_national_scenario_draws.csv',index=False)
ratio_top=(wide['S3']-wide['S1'])/(wide['S2']-wide['S1'])
ratio_exact=(wide['S3']-wide['S1'])/(wide['S8']-wide['S1'])
ratio_summary=pd.DataFrame([{'ratio':'coverage_to_top8_timing_ratio','point':(det.groupby('scenario').deaths_averted.sum()['S3']-det.groupby('scenario').deaths_averted.sum()['S1'])/(det.groupby('scenario').deaths_averted.sum()['S2']-det.groupby('scenario').deaths_averted.sum()['S1']),'median':np.median(ratio_top),'lower_95':np.quantile(ratio_top,.025),'upper_95':np.quantile(ratio_top,.975)},{'ratio':'coverage_to_exact_timing_ratio','point':(det.groupby('scenario').deaths_averted.sum()['S3']-det.groupby('scenario').deaths_averted.sum()['S1'])/(det.groupby('scenario').deaths_averted.sum()['S8']-det.groupby('scenario').deaths_averted.sum()['S1']),'median':np.median(ratio_exact),'lower_95':np.quantile(ratio_exact,.025),'upper_95':np.quantile(ratio_exact,.975)}])
ratio_summary.to_csv(UNC/'source_aligned_decomposition_ratio_summary.csv',index=False)
points=det.groupby('scenario').deaths_averted.sum()
rows=[]
for s in sorted(points.index):
    a=wide[s].to_numpy(float); rows.append({'scenario':s,'point_deaths_averted':points[s],'mc_mean':a.mean(),'mc_median':np.median(a),'lower_95':np.quantile(a,.025),'upper_95':np.quantile(a,.975)})
scenario_summary=pd.DataFrame(rows); scenario_summary.to_csv(UNC/'source_aligned_national_95_intervals.csv',index=False)
contrasts={'S2-S1':('S2','S1'),'S3-S1':('S3','S1'),'S4-S1':('S4','S1'),'S5-S1':('S5','S1'),'S6-S1':('S6','S1'),'S7-S1':('S7','S1'),'S8-S1':('S8','S1')}
rows=[]
for name,(hi,lo) in contrasts.items():
    a=(wide[hi]-wide[lo]).to_numpy(float); rows.append({'contrast':name,'point_incremental_deaths_averted':points[hi]-points[lo],'mc_mean':a.mean(),'mc_median':np.median(a),'lower_95':np.quantile(a,.025),'upper_95':np.quantile(a,.975)})
contrast_summary=pd.DataFrame(rows); contrast_summary.to_csv(UNC/'source_aligned_contrast_95_intervals.csv',index=False)

# Like-for-like 33-province population-average comparator, paired to the same
# mortality and oral-VE draws. Population weighting removes geographic covariance
# between current delivery and model-assigned mortality risk without reintroducing
# Urozgan.
mort33=pd.read_csv(MORT/'provincial_analytic_mortality_weights_33.csv').set_index('province')
pop_w=mort33.u5_population/mort33.u5_population.sum()
s1x=x[x.scenario.eq('S1')].copy()
s1x['population_weight']=s1x.province.map(pop_w)
s1x['impact_fraction_source_aligned']=np.clip(s1x.curve_impact_fraction*s1x.oral_ve_scale,0,1)
avg_frac=s1x.groupby('replicate').apply(lambda g: float((g.impact_fraction_source_aligned*g.population_weight).sum()),include_groups=False).sort_index().to_numpy()
natavg=1610*mort_scale*avg_frac
point_s1=det[det.scenario.eq('S1')].set_index('province')
point_frac=float((point_s1.impact_fraction*pop_w).sum()); point_nat=1610*point_frac
prov=wide['S1'].to_numpy(float)
diff=natavg-prov; rel=100*diff/prov
pd.DataFrame({'replicate':np.arange(N),'mortality_rate_per_100k':mort_rate,'oral_initial_ve':oral_initial_ve,'national_average_deaths_averted':natavg,'provincial_disaggregated_deaths_averted':prov,'difference':diff,'relative_difference_pct':rel}).to_csv(OUT/'source_aligned_national_vs_subnational_draws.csv',index=False)
def ss(name,a,point): return {'quantity':name,'point':point,'mc_mean':np.mean(a),'mc_median':np.median(a),'lower_95':np.quantile(a,.025),'upper_95':np.quantile(a,.975)}
pointr=float(points['S1']); cmp=pd.DataFrame([ss('national_average_deaths_averted',natavg,point_nat),ss('provincial_disaggregated_deaths_averted',prov,pointr),ss('national_minus_provincial_deaths_averted',diff,point_nat-pointr),ss('national_relative_difference_pct',rel,100*(point_nat-pointr)/pointr)])
cmp.to_csv(OUT/'source_aligned_national_vs_subnational_uncertainty.csv',index=False)

print('SOURCE-ALIGNED SCENARIO INTERVALS')
print(scenario_summary.to_string(index=False))
print('\nSOURCE-ALIGNED CONTRAST INTERVALS')
print(contrast_summary.to_string(index=False))
print('\nSOURCE-ALIGNED NATIONAL VS SUBNATIONAL')
print(cmp.to_string(index=False))
print('\nDraw checks: VE median/range95',np.median(oral_initial_ve),np.quantile(oral_initial_ve,[.025,.975]),'mortality rate median/range95',np.median(mort_rate),np.quantile(mort_rate,[.025,.975]))
