#!/usr/bin/env python3
"""Reporting outputs requested during co-author meetings.

Creates data tables and publication-oriented figures without changing model inputs.
"""
from pathlib import Path
import json, sys, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from rotavirus_impact.impact_model import (
    ORAL_ROTARIX, impact_fraction, load_curve_shapes, disease_age_mass,
    protection_profile, CURVE_ENDPOINT_DAYS, U5_DAYS
)

OUT=ROOT/'outputs'/'reporting'; FIG=OUT/'figures'; ENV=FIG/'province_envelopes'
OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True); ENV.mkdir(parents=True,exist_ok=True)
CUR=ROOT/'outputs'/'coverage_curves'; MORT=ROOT/'outputs'/'mortality'; IMP=ROOT/'outputs'/'impact'; UNC=ROOT/'outputs'/'uncertainty_sensitivity'

SIX=['BAMYAN','DAYKUNDI','KABUL','KANDAHAR','HELMAND','UROZGAN']

def label(x):
    return x.title().replace('Kunarha','Kunar').replace('Nooristan','Nuristan').replace('Sar-E-Pul','Sar-e-Pul').replace('Maidan Wardak','Maidan Wardak')

def national_vs_subnational():
    """Compare population-averaged delivery with province-specific delivery.

    Both calculations use the same 33 analytic provinces. The national-average
    comparator weights provincial current-programme impact fractions by under-five
    population and applies the resulting common delivery profile to the fixed
    1,610-death reference. This removes geographic covariance between delivery
    and model-assigned mortality risk without reintroducing Urozgan.
    """
    det=pd.read_csv(IMP/'provincial_primary_s1_s8.csv')
    s1=det[det.scenario.eq('S1')].set_index('province')
    mort=pd.read_csv(MORT/'provincial_analytic_mortality_weights_33.csv').set_index('province')
    weights=mort.u5_population/mort.u5_population.sum()
    nat_frac=float((s1.impact_fraction*weights).sum())
    nat_averted=1610.0*nat_frac
    sens=pd.read_csv(UNC/'national_mortality_weight_sensitivity.csv')
    mapping={'uniform':'uniform','composite_primary_linear_inverse':'linear_inverse_primary','composite_log_linear':'log_linear','composite_bounded_2x':'bounded_2x','composite_bounded_5x':'bounded_5x','ors_adjusted':'ors_adjusted','care_seeking_adjusted':'care_seeking_adjusted','underweight_adjusted':'underweight_adjusted'}
    rows=[]
    for k,name in mapping.items():
        z=sens[(sens.mortality_weighting==k)&(sens.scenario=='S1')].iloc[0]
        prov=float(z.deaths_averted)
        rows.append({'mortality_weighting':name,'national_average_model_deaths_averted':nat_averted,'provincial_disaggregated_deaths_averted':prov,'national_minus_provincial':nat_averted-prov,'relative_difference_vs_provincial_pct':(nat_averted-prov)/prov*100})
    out=pd.DataFrame(rows); out.to_csv(OUT/'national_vs_subnational_impact.csv',index=False)
    q=out[out.mortality_weighting=='linear_inverse_primary'].iloc[0]
    fig,ax=plt.subplots(figsize=(7.2,4.8))
    ax.bar(['33-province population-average\ndelivery','Province-specific\ndelivery'],[q.national_average_model_deaths_averted,q.provincial_disaggregated_deaths_averted])
    ax.set_ylabel('Rotavirus deaths averted per year')
    ax.set_title('Population-average versus province-specific impact')
    ax.text(0.5,max(q.national_average_model_deaths_averted,q.provincial_disaggregated_deaths_averted)*0.94,f'Primary allocation difference: {q.relative_difference_vs_provincial_pct:.1f}%',ha='center')
    fig.tight_layout(); fig.savefig(FIG/'figure_national_vs_subnational.png',dpi=600,bbox_inches='tight'); fig.savefig(FIG/'figure_national_vs_subnational.pdf',bbox_inches='tight'); plt.close(fig)
    return out

def six_province_visual():
    cov=pd.read_csv(ROOT/'outputs'/'audit'/'mics_provincial_coverage_audit.csv')
    mort=pd.read_csv(MORT/'provincial_care_access_and_mortality_weights.csv')
    x=cov.merge(mort[['province','cfr_multiplier_composite','baseline_deaths_composite','care_access_composite_z']],on='province',how='left')
    x=x[x.province.isin(SIX)].copy()
    x['province_label']=x.province.map(label)
    # order best-to-worst current RV1
    x=x.sort_values('rv1_weighted_coverage',ascending=False)
    x.to_csv(OUT/'six_province_coverage_mortality.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5.2))
    idx=np.arange(len(x)); w=0.34
    ax.bar(idx-w/2,x.rv1_weighted_coverage*100,w,label='RV1 coverage')
    ax.bar(idx+w/2,x.rv2_weighted_coverage*100,w,label='RV2 coverage')
    ax.set_ylabel('Vaccination coverage (%)'); ax.set_xticks(idx); ax.set_xticklabels(x.province_label,rotation=25,ha='right')
    ax.set_ylim(0,105); ax.legend(loc='upper left')
    ax2=ax.twinx(); ax2.plot(idx,x.cfr_multiplier_composite,marker='o',label='Relative CFR multiplier')
    ax2.set_ylabel('Relative CFR multiplier (population-normalised)')
    ax.set_title('Coverage gradient and mortality-risk allocation in six illustrative provinces')
    fig.tight_layout(); fig.savefig(FIG/'figure_six_province_coverage_mortality.png',dpi=600,bbox_inches='tight'); fig.savefig(FIG/'figure_six_province_coverage_mortality.pdf',bbox_inches='tight'); plt.close(fig)
    return x

def envelope_data():
    fits=pd.read_csv(CUR/'coverage_curve_fits.csv'); curves=pd.read_csv(CUR/'coverage_curves_daily.csv'); shapes=load_curve_shapes(curves,fits)
    mort34=pd.read_csv(MORT/'provincial_care_access_and_mortality_weights.csv').set_index('province')
    f=fits.set_index(['province','dose'])
    disease=disease_age_mass('burr_primary')
    rows=[]
    for province in sorted(set(mort34.index) - {'UROZGAN'}):
        c1=float(f.loc[(province,'RV1'),'final_crude_coverage']); c2=float(f.loc[(province,'RV2'),'final_crude_coverage'])
        sh1=shapes[(province,'RV1')]; sh2=shapes[(province,'RV2')]
        base=float(mort34.loc[province,'baseline_deaths_composite'])
        # Middle envelope follows Andy Clark's verbal conceptual decomposition:
        # burden in the fraction of children ultimately reached by >=RV1. Timing is
        # represented only in the inner prevented envelope so the middle layer remains a reach construct
        # that marginal RV1/RV2 curves form a perfectly nested joint process.
        rv1_reached=np.full(U5_DAYS,c1,dtype=float)
        prot=protection_profile(rv1_final=c1,rv2_final=c2,rv1_timing_shape=sh1,rv2_timing_shape=sh2,product=ORAL_ROTARIX,horizon_days=U5_DAYS)
        daily_total=base*disease; daily_reached=daily_total*rv1_reached; daily_prevented=daily_total*prot
        if np.any(daily_prevented-daily_reached>1e-10): raise AssertionError(province+' prevented exceeds reached envelope')
        cum_total=np.cumsum(daily_total); cum_reached=np.cumsum(daily_reached); cum_prevented=np.cumsum(daily_prevented)
        for day in range(U5_DAYS):
            rows.append({'province':province,'age_days':day,'age_weeks':day/7,'cum_total_preventable_deaths':cum_total[day],'cum_burden_in_eventually_rv1_reached':cum_reached[day],'cum_deaths_prevented_current':cum_prevented[day]})
        fig,ax=plt.subplots(figsize=(6.6,4.5))
        age=np.arange(U5_DAYS)/7
        ax.fill_between(age,0,cum_total,alpha=.22,label='Total rotavirus deaths (opportunity envelope)')
        ax.fill_between(age,0,cum_reached,alpha=.34,label='Burden among children ultimately reached by RV1')
        ax.fill_between(age,0,cum_prevented,alpha=.48,label='Deaths prevented under observed delivery')
        ax.set_xlim(0,104); ax.set_xlabel('Age (weeks)'); ax.set_ylabel('Cumulative annual deaths')
        ax.set_title(label(province)); ax.legend(fontsize=8,loc='center left',bbox_to_anchor=(1.02,0.5)); fig.tight_layout()
        fig.savefig(ENV/f'{province.lower().replace(" ","_")}_envelope.png',dpi=450,bbox_inches='tight'); plt.close(fig)
    # descriptive Urozgan can still have envelope based 34-p burden because its timing fit exists; mark descriptive only
    province='UROZGAN'
    if (province,'RV1') in f.index and (province,'RV1') in shapes:
        c1=float(f.loc[(province,'RV1'),'final_crude_coverage']); c2=float(f.loc[(province,'RV2'),'final_crude_coverage']); sh1=shapes[(province,'RV1')]; sh2=shapes[(province,'RV2')]
        base=float(mort34.loc[province,'baseline_deaths_composite']); rv1_reached=np.full(U5_DAYS,c1,dtype=float)
        prot=protection_profile(rv1_final=c1,rv2_final=c2,rv1_timing_shape=sh1,rv2_timing_shape=sh2,product=ORAL_ROTARIX,horizon_days=U5_DAYS)
        daily_total=base*disease; cum_total=np.cumsum(daily_total); cum_reached=np.cumsum(daily_total*rv1_reached); cum_prevented=np.cumsum(daily_total*prot)
        for day in range(U5_DAYS): rows.append({'province':province,'age_days':day,'age_weeks':day/7,'cum_total_preventable_deaths':cum_total[day],'cum_burden_in_eventually_rv1_reached':cum_reached[day],'cum_deaths_prevented_current':cum_prevented[day]})
        fig,ax=plt.subplots(figsize=(6.6,4.5)); age=np.arange(U5_DAYS)/7
        ax.fill_between(age,0,cum_total,alpha=.22,label='Total rotavirus deaths (opportunity envelope)'); ax.fill_between(age,0,cum_reached,alpha=.34,label='Burden among children ultimately reached by RV1'); ax.fill_between(age,0,cum_prevented,alpha=.48,label='Deaths prevented under observed delivery')
        ax.set_xlim(0,104); ax.set_xlabel('Age (weeks)'); ax.set_ylabel('Cumulative annual deaths'); ax.set_title('Urozgan (descriptive only)'); ax.legend(fontsize=8,loc='center left',bbox_to_anchor=(1.02,0.5)); fig.tight_layout(); fig.savefig(ENV/'urozgan_envelope_descriptive.png',dpi=450,bbox_inches='tight'); plt.close(fig)
    out=pd.DataFrame(rows); out.to_csv(OUT/'province_layered_envelope_daily.csv',index=False)
    return out

def provincial_ci_figure():
    ci=pd.read_csv(UNC/'provincial_primary_95_intervals.csv'); s1=ci[ci.scenario=='S1'].sort_values('point_deaths_averted')
    fig,ax=plt.subplots(figsize=(8.3,9.0)); y=np.arange(len(s1))
    lo=s1.point_deaths_averted-s1.lower_95; hi=s1.upper_95-s1.point_deaths_averted
    ax.errorbar(s1.point_deaths_averted,y,xerr=np.vstack([lo,hi]),fmt='o',capsize=2)
    ax.set_yticks(y); ax.set_yticklabels(s1.province.map(label),fontsize=7.5); ax.set_xlabel('S1 deaths averted per year (95% Monte Carlo interval)'); ax.set_title('Provincial observed-delivery impact and uncertainty'); fig.tight_layout(); fig.savefig(FIG/'figure_provincial_s1_intervals.png',dpi=600,bbox_inches='tight'); fig.savefig(FIG/'figure_provincial_s1_intervals.pdf',bbox_inches='tight'); plt.close(fig)

def decomposition_figure():
    ci=pd.read_csv(UNC/'national_contrast_95_intervals.csv'); ci=ci[ci.uncertainty_structure=='province_independent_plus_curve_fit_parameters'].copy()
    order=['S2-S1','S3-S1','S6-S1','S8-S1']
    names={'S2-S1':'Timing: top-8 pattern','S3-S1':'Coverage: top-8 mean','S6-S1':'Product: 94% non-waning','S8-S1':'Timing: exact 6/10 weeks'}
    x=ci[ci.contrast.isin(order)].set_index('contrast').loc[order].reset_index(); y=np.arange(len(x))
    fig,ax=plt.subplots(figsize=(8,4.8)); lo=x.point_incremental_deaths_averted-x.lower_95; hi=x.upper_95-x.point_incremental_deaths_averted
    ax.errorbar(x.point_incremental_deaths_averted,y,xerr=np.vstack([lo,hi]),fmt='o',capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([names[v] for v in x.contrast]); ax.set_xlabel('Additional deaths averted per year vs observed delivery'); ax.set_title('Coverage, timing, and product opportunity gaps'); fig.tight_layout(); fig.savefig(FIG/'figure_decomposition_with_intervals.png',dpi=600,bbox_inches='tight'); fig.savefig(FIG/'figure_decomposition_with_intervals.pdf',bbox_inches='tight'); plt.close(fig)

def five_weighting_wide():
    x=pd.read_csv(UNC/'national_mortality_weight_sensitivity.csv')
    keep={'uniform':'Uniform','composite_primary_linear_inverse':'Composite (primary)','ors_adjusted':'ORS-adjusted','care_seeking_adjusted':'Care-seeking-adjusted','underweight_adjusted':'Underweight-adjusted'}
    x=x[x.mortality_weighting.isin(keep)].copy(); x['mortality_weighting']=x.mortality_weighting.map(keep)
    wide=x.pivot(index='scenario',columns='mortality_weighting',values='deaths_averted').reset_index()
    wide.to_csv(OUT/'five_mortality_weighting_sensitivity_side_by_side.csv',index=False)
    return wide

def main():
    nv=national_vs_subnational(); six=six_province_visual(); env=envelope_data(); provincial_ci_figure(); decomposition_figure(); five=five_weighting_wide()
    audit={'national_vs_subnational_primary':nv[nv.mortality_weighting=='linear_inverse_primary'].iloc[0].to_dict(),'six_provinces':SIX,'envelope_provinces':int(env.province.nunique()),'individual_envelope_png_count':len(list(ENV.glob('*.png'))),'five_weighting_columns':five.columns.tolist()}
    (OUT/'reporting_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8'); print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
