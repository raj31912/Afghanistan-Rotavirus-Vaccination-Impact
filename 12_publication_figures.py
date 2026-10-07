#!/usr/bin/env python3
"""Generate the journal-facing Figure 1 and Figure 6 panels from derived outputs."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from rotavirus_impact.impact_model import disease_age_mass, protection_profile, load_curve_shapes, ORAL_ROTARIX

CUR = ROOT / "outputs" / "coverage_curves"
MORT = ROOT / "outputs" / "mortality"
FIG = ROOT / "figures"
MAIN = FIG / "main"
SUPP = FIG / "supplementary"
SRC = FIG / "source_data"
for p in (MAIN, SUPP, SRC): p.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":9,"axes.labelsize":10,
    "xtick.labelsize":9,"ytick.labelsize":9,"axes.titlesize":10.5,
    "axes.titleweight":"bold","axes.spines.top":False,"axes.spines.right":False,
    "axes.linewidth":0.8,"grid.alpha":0.25,
})
BLUE="#2b6cb0"; RED="#c53030"; GREEN="#38a169"; GREY="#718096"
PURPLE="#805ad5"; AMBER="#d69e2e"; DARKGREY="#4a5568"
LIGHT_GREY="#e2e8f0"; MID_BLUE="#90cdf4"
FOCAL=["BAMYAN","DAYKUNDI","KABUL","KANDAHAR","HELMAND","UROZGAN"]
COLORS={"BAMYAN":BLUE,"DAYKUNDI":GREEN,"KABUL":PURPLE,"KANDAHAR":AMBER,"HELMAND":RED,"UROZGAN":DARKGREY}

def figure1():
    curves=pd.read_csv(CUR/"coverage_curves_daily.csv")
    fits=pd.read_csv(CUR/"coverage_curve_fits.csv")
    curves=curves[curves.province.ne("AFGHANISTAN")].copy(); curves["age_weeks"]=curves.age_days/7; curves["coverage_pct"]=curves.fitted_rescaled*100
    fits=fits[fits.province.ne("AFGHANISTAN")].copy(); fits["final_coverage_pct"]=fits.final_crude_coverage*100; fits["median_age_weeks"]=fits.timing_unweighted_median_age_days/7

    fig,axs=plt.subplots(2,3,figsize=(12.8,8),sharex=True,sharey=True); axs=axs.ravel()
    for ax,province in zip(axs,FOCAL):
        for dose,ls in (("RV1","-"),("RV2","--")):
            d=curves[(curves.province==province)&(curves.dose==dose)]
            ax.plot(d.age_weeks,d.coverage_pct,color=COLORS[province],lw=2.1,ls=ls)
        ax.axvline(6,color=GREY,ls=":",lw=1); ax.axvline(10,color=GREY,ls=":",lw=1)
        r1=float(fits[(fits.province==province)&(fits.dose=="RV1")].final_coverage_pct.iloc[0]); r2=float(fits[(fits.province==province)&(fits.dose=="RV2")].final_coverage_pct.iloc[0])
        ax.set_title(f"{province.title()}  (RV1 {r1:.1f}%, RV2 {r2:.1f}%)",loc="left"); ax.set_xlim(0,104); ax.set_ylim(0,100); ax.grid(True)
    for ax in axs[3:]: ax.set_xlabel("Age (weeks)")
    for ax in axs[::3]: ax.set_ylabel("Cumulative coverage (%)")
    handles=[Line2D([0],[0],color="black",lw=2.1,ls="-",label="RV1"),Line2D([0],[0],color="black",lw=2.1,ls="--",label="RV2"),Line2D([0],[0],color=GREY,lw=1,ls=":",label="6 and 10 weeks")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(0.5,0.01),ncol=3,frameon=False); fig.tight_layout(rect=[0,0.05,1,1])
    fig.savefig(MAIN/"Figure1A_Provincial_Coverage_Curves.png",dpi=300,bbox_inches="tight"); fig.savefig(MAIN/"Figure1A_Provincial_Coverage_Curves.pdf",bbox_inches="tight"); plt.close(fig)
    curves[curves.province.isin(FOCAL)][["province","dose","age_days","age_weeks","coverage_pct"]].to_csv(SRC/"Figure1A_Provincial_Coverage_Curves_SourceData.csv",index=False)

    fig,ax=plt.subplots(figsize=(9.6,6.4))
    for province,d in fits.groupby("province"):
        if set(d.dose)=={"RV1","RV2"}:
            d=d.sort_values("dose"); ax.plot(d.final_coverage_pct,d.median_age_weeks,color="#CBD5E0",lw=.8,zorder=1)
    rv1=fits[fits.dose=="RV1"]; rv2=fits[fits.dose=="RV2"]
    ax.scatter(rv1.final_coverage_pct,rv1.median_age_weeks,s=42,color=BLUE,alpha=.9,label="RV1",zorder=3)
    ax.scatter(rv2.final_coverage_pct,rv2.median_age_weeks,s=48,color=RED,marker="s",alpha=.9,label="RV2",zorder=3)
    for province in FOCAL:
        for dose,color,dy in (("RV1",BLUE,.35),("RV2",RED,-.45)):
            d=fits[(fits.province==province)&(fits.dose==dose)]
            if len(d): ax.text(float(d.final_coverage_pct.iloc[0])+1.2,float(d.median_age_weeks.iloc[0])+dy,province.title(),fontsize=8.3,color=color)
    ax.axhline(6,color=GREY,ls=":",lw=1); ax.axhline(10,color=GREY,ls=":",lw=1)
    ax.text(3,6.2,"6-week target",color=GREY,fontsize=8); ax.text(3,10.2,"10-week target",color=GREY,fontsize=8)
    ax.set_xlabel("Final cumulative coverage (%)"); ax.set_ylabel("Median vaccination age (weeks)"); ax.set_xlim(0,102); ax.set_ylim(0,max(fits.median_age_weeks.max()+2,22)); ax.grid(True); ax.legend(frameon=False,loc="upper right"); fig.tight_layout()
    fig.savefig(MAIN/"Figure1B_Coverage_vs_MedianAge.png",dpi=300,bbox_inches="tight"); fig.savefig(MAIN/"Figure1B_Coverage_vs_MedianAge.pdf",bbox_inches="tight"); plt.close(fig)
    out=fits[["province","dose","final_coverage_pct","median_age_weeks"]].copy(); out["is_focal"]=out.province.isin(FOCAL); out.to_csv(SRC/"Figure1B_Coverage_vs_MedianAge_SourceData.csv",index=False)

def figure6():
    curves=pd.read_csv(CUR/"coverage_curves_daily.csv")
    fits=pd.read_csv(CUR/"coverage_curve_fits.csv")
    m33=pd.read_csv(MORT/"provincial_analytic_mortality_weights_33.csv").set_index("province")
    m34=pd.read_csv(MORT/"provincial_care_access_and_mortality_weights.csv").set_index("province")
    shapes=load_curve_shapes(curves,fits); f=fits.set_index(["province","dose"]); disease=disease_age_mass("burr_primary")
    rows=[]
    provinces=sorted(p for p in fits.province.unique() if p!="AFGHANISTAN")
    for province in provinces:
        base=float(m34.loc[province,"baseline_deaths_composite"] if province=="UROZGAN" else m33.loc[province,"baseline_deaths_composite"])
        c1=float(f.loc[(province,"RV1"),"final_crude_coverage"]); c2=float(f.loc[(province,"RV2"),"final_crude_coverage"])
        prot=protection_profile(rv1_final=c1,rv2_final=c2,rv1_timing_shape=shapes[(province,"RV1")],rv2_timing_shape=shapes[(province,"RV2")],product=ORAL_ROTARIX,horizon_days=len(disease))
        # Programme reach is defined by eventual RV1 receipt, independent of
        # vaccination timing. The middle envelope is therefore the fixed share
        # of age-specific burden among children eventually reached by RV1.
        total=base*disease; reachable=total*c1; prevented=total*prot
        for day in range(min(len(disease),731)):
            rows.append({"province":province,"age_days":day,"age_months":day/30.4375,"total_deaths_density":total[day]*7,"reachable_deaths_density":reachable[day]*7,"prevented_deaths_density":prevented[day]*7,"baseline_deaths":base})
    d=pd.DataFrame(rows); d.to_csv(SRC/"Figure6_Envelopes_SourceData.csv",index=False)
    ymax=d[d.province.isin(FOCAL)].total_deaths_density.max()*1.08
    handles=[Patch(facecolor=LIGHT_GREY,alpha=.6,label="Total burden"),Patch(facecolor=MID_BLUE,alpha=.7,label="Burden among children eventually reached by RV1"),Patch(facecolor=BLUE,alpha=.9,label="Deaths prevented under observed delivery")]
    fig,axs=plt.subplots(2,3,figsize=(14.5,8.5),sharex=True,sharey=True); axs=axs.ravel()
    for ax,province in zip(axs,FOCAL):
        q=d[d.province==province]; x=q.age_months.to_numpy(); ax.fill_between(x,0,q.total_deaths_density,color=LIGHT_GREY,alpha=.6); ax.fill_between(x,0,q.reachable_deaths_density,color=MID_BLUE,alpha=.7); ax.fill_between(x,0,q.prevented_deaths_density,color=BLUE,alpha=.9)
        base=float(q.baseline_deaths.iloc[0]); reach=q.reachable_deaths_density.sum()/q.total_deaths_density.sum()*100; prev=q.prevented_deaths_density.sum()/q.total_deaths_density.sum()*100
        ax.set_title(f"{province.title()} — {base:.1f} baseline deaths/year",loc="left"); ax.text(.97,.92,f"{reach:.0f}% eventually reached\n{prev:.0f}% prevented",transform=ax.transAxes,ha="right",va="top",fontsize=8.3,bbox=dict(facecolor="white",edgecolor="none",alpha=.75,pad=1.5)); ax.set_xlim(0,24); ax.set_ylim(0,ymax); ax.grid(True); ax.set_xticks([0,6,12,18,24])
    for ax in axs[3:]: ax.set_xlabel("Age (months)")
    for ax in axs[::3]: ax.set_ylabel("Annual deaths per week of age")
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,-.01),ncol=3,frameon=False); fig.tight_layout(rect=[0,.05,1,1]); fig.savefig(MAIN/"Figure6_Six_Province_Envelopes.png",dpi=600,bbox_inches="tight"); fig.savefig(MAIN/"Figure6_Six_Province_Envelopes.pdf",bbox_inches="tight"); plt.close(fig)

    fig,axs=plt.subplots(6,6,figsize=(18,14),sharex=True,sharey=True); axs=axs.ravel(); ymax=d.total_deaths_density.max()*1.05
    for ax,province in zip(axs,provinces):
        q=d[d.province==province]; x=q.age_months.to_numpy(); ax.fill_between(x,0,q.total_deaths_density,color=LIGHT_GREY,alpha=.6); ax.fill_between(x,0,q.reachable_deaths_density,color=MID_BLUE,alpha=.7); ax.fill_between(x,0,q.prevented_deaths_density,color=BLUE,alpha=.9); ax.set_title(f"{province.title()} — {float(q.baseline_deaths.iloc[0]):.1f}",loc="left",fontsize=7); ax.set_xlim(0,24); ax.set_ylim(0,ymax); ax.grid(True); ax.tick_params(labelsize=6,length=2)
    for ax in axs[len(provinces):]: ax.axis("off")
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,-.005),ncol=3,frameon=False,fontsize=8); fig.tight_layout(rect=[0,.04,1,1]); fig.savefig(SUPP/"FigureG6_All_Provinces_Envelopes.pdf",bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    figure1(); figure6()
