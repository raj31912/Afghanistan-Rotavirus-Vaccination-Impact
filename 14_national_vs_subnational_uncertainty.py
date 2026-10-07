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
from rotavirus_impact.uncertainty_sensitivity import MonteCarloSettings,simulate_provincial_independent,effective_ve_scale

CUR=ROOT/'outputs'/'coverage_curves'; IMP=ROOT/'outputs'/'impact'; UNC=ROOT/'outputs'/'uncertainty_sensitivity'; OUT=ROOT/'outputs'/'reporting'; OUT.mkdir(exist_ok=True)
settings=MonteCarloSettings(replicates=500,seed=SEED)
fits=pd.read_csv(CUR/'coverage_curve_fits.csv')
curves=pd.read_csv(CUR/'coverage_curves_daily.csv')
shapes=load_curve_shapes(curves,fits)
f= fits.set_index(['province','dose'])
c1=float(f.loc[('AFGHANISTAN','RV1'),'final_crude_coverage']); c2=float(f.loc[('AFGHANISTAN','RV2'),'final_crude_coverage'])
point_frac=impact_fraction(rv1_final=c1,rv2_final=c2,rv1_timing_shape=shapes[('AFGHANISTAN','RV1')],rv2_timing_shape=shapes[('AFGHANISTAN','RV2')],product=ORAL_ROTARIX,age_distribution='burr_primary')
point_nat=1610*point_frac

# Conditional uncertainty in the nationally fitted timing curves: 100 whole-national curve realisations.
rng=np.random.default_rng(SEED+1414)
idx=fits[fits.province.eq('AFGHANISTAN')].set_index('dose')
curve_fracs=[]
diag={'parameter_draws':0,'clipped_fallbacks':0,'total_rejection_attempts':0,'curve_realisations':100}
for d in range(100):
    p1,i1=draw_params(idx.loc['RV1'],rng); p2,i2=draw_params(idx.loc['RV2'],rng)
    sh1=timing_shape_from_params(p1,RV1_TARGET_DAYS); sh2=timing_shape_from_params(p2,RV2_TARGET_DAYS)
    curve_fracs.append(impact_fraction(rv1_final=c1,rv2_final=c2,rv1_timing_shape=sh1,rv2_timing_shape=sh2,product=ORAL_ROTARIX,age_distribution='burr_primary'))
    for info in (i1,i2):
        diag['parameter_draws']+=1; diag['clipped_fallbacks']+=int(info['clipped_fallback']); diag['total_rejection_attempts']+=int(info['attempts']-1)
curve_fracs=np.asarray(curve_fracs)
selected=rng.integers(0,100,size=settings.replicates)
frac_rep=curve_fracs[selected]

# Use the same province-specific mortality/VE shocks as the specified primary
# provincial uncertainty model. Applying the national-average delivery fraction to
# each shocked provincial burden gives a paired aggregation comparison: common
# mortality/VE shocks cancel appropriately when national and province-summed impact
# are contrasted within replicate.
det=pd.read_csv(IMP/'provincial_primary_s1_s8.csv')
ind=simulate_provincial_independent(det,settings)
shock=ind[['province','replicate','mortality_scale','ve_scale_raw']].drop_duplicates()
base=det[det.scenario.eq('S1')].set_index('province')['baseline_deaths']
shock['baseline_point']=shock.province.map(base)
shock['baseline_deaths']=shock.baseline_point*shock.mortality_scale
shock['ve_eff']=effective_ve_scale(shock.ve_scale_raw.to_numpy(),ORAL_ROTARIX.name)
shock['curve_frac']=shock.replicate.map(pd.Series(frac_rep,index=np.arange(settings.replicates)))
shock['national_average_averted_component']=shock.baseline_deaths*shock.ve_eff*shock.curve_frac
nat_draw=shock.groupby('replicate')['national_average_averted_component'].sum().rename('national_average_deaths_averted').reset_index()
prov_wide=pd.read_csv(UNC/'primary_national_scenario_draws.csv')
prov_draw=prov_wide[['replicate','S1']].rename(columns={'S1':'provincial_disaggregated_deaths_averted'})
x=nat_draw.merge(prov_draw,on='replicate',validate='one_to_one')
x['difference_deaths_averted']=x.national_average_deaths_averted-x.provincial_disaggregated_deaths_averted
x['relative_difference_pct']=100*x.difference_deaths_averted/x.provincial_disaggregated_deaths_averted
x.to_csv(OUT/'national_vs_subnational_mc_draws.csv',index=False)

def summ(col):
    a=x[col].to_numpy(float)
    return {'mc_mean':a.mean(),'mc_median':np.median(a),'lower_95':np.quantile(a,.025),'upper_95':np.quantile(a,.975)}
point_prov=float(det[det.scenario.eq('S1')].deaths_averted.sum())
summary=pd.DataFrame([
    {'quantity':'national_average_deaths_averted','point':point_nat,**summ('national_average_deaths_averted')},
    {'quantity':'provincial_disaggregated_deaths_averted','point':point_prov,**summ('provincial_disaggregated_deaths_averted')},
    {'quantity':'national_minus_provincial_deaths_averted','point':point_nat-point_prov,**summ('difference_deaths_averted')},
    {'quantity':'national_relative_difference_pct','point':100*(point_nat-point_prov)/point_prov,**summ('relative_difference_pct')},
])
summary.to_csv(OUT/'national_vs_subnational_uncertainty.csv',index=False)
print('National average point fraction',point_frac)
print(summary.to_string(index=False))
print('Curve uncertainty diagnostics',diag)
