import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9a9a92','axes.labelcolor':'#3d3d3a','xtick.color':'#5f5f5a','ytick.color':'#5f5f5a'})
rel=pd.read_csv('results/relative_MAE_vs_naive.csv')
M=[('sir_ml_ww','Hybrid SIR-IVP (with wastewater)','#2a78d6','o'),('sir_ml','Hybrid SIR-IVP (hospital data only)','#eb6834','s'),
   ('pure_ml_ww','Pure ML + wastewater (no SIR)','#1baf7a','^'),('sir_ar','SIR-IVP + linear AR','#eda100','D'),('sir_const','SIR-IVP, constant β','#e87ba4','v')]
fig,ax=plt.subplots(1,2,figsize=(10,4),sharey=True)
for a,(p,t) in zip(ax,[('covid','COVID-19'),('flu','Influenza')]):
    r=rel[rel.pathogen==p]
    a.axhline(1,color='#9a9a92',lw=1,ls='--'); a.text(4.12,1.03,'naive',va='center',color='#5f5f5a',fontsize=9)
    for m,l,c,mk in M: a.plot(r.h,r[m],color=c,lw=2,marker=mk,ms=7,label=l,markeredgecolor='white',markeredgewidth=1.5)
    a.set_title(t,loc='left',color='#1f1f1d')
    a.set_xticks([1,2,3,4]); a.set_xlabel('Forecast horizon (weeks)'); a.set_ylim(0.6,1.6)
    if p=='flu': a.text(2.55,1.55,'constant β off scale\n(2.28 at h=3, 4.19 at h=4)',fontsize=8,color='#5f5f5a')
    a.grid(axis='y',color='#e6e5e0',lw=0.8)
ax[0].set_ylabel('Relative MAE vs naive\n(geometric mean, 10 states)')
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=3,frameon=False,fontsize=9,bbox_to_anchor=(0.5,-0.06))
fig.suptitle('Out-of-sample forecast error, Jan 2025 – Oct 2026 (lower is better)',x=0.01,ha='left',fontsize=11,color='#1f1f1d')
fig.tight_layout(rect=(0,0.08,1,0.95)); fig.savefig('results/fig1_relative_mae.png',dpi=200,bbox_inches='tight')
F=pd.read_csv('results/forecasts.csv',parse_dates=['origin']); F=F[(F.h==2)&(F.model=='sir_ml_ww')]
F['target']=F.origin+pd.Timedelta(weeks=2)
fig,ax=plt.subplots(2,2,figsize=(10,6))
for i,p in enumerate(['covid','flu']):
    for j,st in enumerate(['CA','NY']):
        g=F[(F.pathogen==p)&(F.state==st)].sort_values('target'); a=ax[i,j]
        a.plot(g.target,g.obs,color='#3d3d3a',lw=2,label='Observed admissions')
        a.plot(g.target,g.pred,color='#2a78d6',lw=2,ls='-',label='Hybrid SIR-IVP (with wastewater), 2 weeks ahead')
        a.set_title(f"{'COVID-19' if p=='covid' else 'Influenza'} — {st}",loc='left',fontsize=10,color='#1f1f1d')
        a.grid(axis='y',color='#e6e5e0',lw=0.8); a.tick_params(axis='x',labelsize=8)
        if j==0: a.set_ylabel('Weekly admissions')
h,l=ax[0,0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig('results/fig2_forecast_examples.png',dpi=200,bbox_inches='tight')

# ---- Fig 3: simulated outages ----
rel=pd.read_csv('results/outage_relMAE_vs_locf.csv')
M3=[('sir_ww','Hybrid SIR-IVP (with wastewater)','#2a78d6','o'),('sir_ml','Hybrid SIR-IVP (hospital data only)','#eb6834','s'),
    ('ww_ratio','Wastewater ratio scaling (no SIR)','#1baf7a','^'),('sir_const','SIR-IVP, constant β','#e87ba4','v')]
fig,ax=plt.subplots(1,2,figsize=(10,4),sharey=True)
for a,(p,t) in zip(ax,[('covid','COVID-19'),('flu','Influenza')]):
    r=rel[rel.pathogen==p]
    a.axhline(1,color='#9a9a92',lw=1,ls='--'); a.text(8.15,1.03,'last value\ncarried forward',color='#5f5f5a',fontsize=8)
    for m,l,c,mk in M3: a.plot(r.k,r[m],color=c,lw=2,marker=mk,ms=7,label=l,markeredgecolor='white',markeredgewidth=1.5)
    a.set_title(t,loc='left',color='#1f1f1d'); a.set_xticks(range(1,9)); a.set_xlabel('Weeks since hospital reporting stopped'); a.set_ylim(0.2,2.0)
    a.grid(axis='y',color='#e6e5e0',lw=0.8)
ax[0].set_ylabel('Relative MAE vs last value\n(geometric mean, 10 states)')
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=2,frameon=False,fontsize=9,bbox_to_anchor=(0.5,-0.08))
fig.suptitle('Simulated 8-week hospital-reporting outages, 2025–2026 (lower is better; constant β leaves the scale after week 3)',x=0.01,ha='left',fontsize=10.5,color='#1f1f1d')
fig.tight_layout(rect=(0,0.08,1,0.95)); fig.savefig('results/fig3_outage.png',dpi=200,bbox_inches='tight')

# ---- Fig 4: real 2024 voluntary-reporting gap, COVID-19 ----
B=pd.read_csv('results/real_gap_2024.csv',parse_dates=['week_end']); B=B[B.pathogen=='covid']; B['proxy']=B.reported/B.reporting_frac
fig,ax=plt.subplots(2,2,figsize=(10,6))
for a,st in zip(ax.flat,['NY','IL','OH','TX']):
    g=B[B.state==st]
    a.plot(g.week_end,g.proxy,color='#3d3d3a',lw=2,label='Reported ÷ fraction of hospitals reporting (proxy)')
    a.plot(g.week_end,g.sir_ww,color='#2a78d6',lw=2,label='Hybrid SIR-IVP (with wastewater)')
    a.plot(g.week_end,g.sir_ml,color='#eb6834',lw=2,ls='--',label='Hybrid SIR-IVP (hospital data only)')
    a.set_title(st,loc='left',fontsize=10,color='#1f1f1d'); a.grid(axis='y',color='#e6e5e0',lw=0.8); a.tick_params(axis='x',labelsize=8)
for a in ax[:,0]: a.set_ylabel('Weekly COVID-19 admissions')
h,l=ax[0,0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=3,frameon=False,fontsize=8.5,bbox_to_anchor=(0.5,-0.02))
fig.suptitle('COVID-19 during the May–Oct 2024 voluntary-reporting period (models trained on data to 4 May 2024)',x=0.01,ha='left',fontsize=10.5,color='#1f1f1d')
fig.tight_layout(rect=(0,0.05,1,0.96)); fig.savefig('results/fig4_real_gap_2024.png',dpi=200,bbox_inches='tight')

# ---- Fig 5: probabilistic forecasts (fan chart), hybrid SIR-IVP with wastewater, 2 weeks ahead ----
P=pd.read_csv('results/probabilistic_forecasts.csv',parse_dates=['origin'])
P=P[(P.model=='sir_ml_ww')&(P.h==2)].copy(); P['target']=P.origin+pd.Timedelta(weeks=2)
fig,ax=plt.subplots(2,2,figsize=(10,6))
for i,p in enumerate(['covid','flu']):
    for j,st in enumerate(['CA','NY']):
        g=P[(P.pathogen==p)&(P.state==st)].sort_values('target'); a=ax[i,j]
        a.fill_between(g.target,g['lo0.05'],g['hi0.05'],color='#2a78d6',alpha=0.15,lw=0,label='95% interval')
        a.fill_between(g.target,g['lo0.5'],g['hi0.5'],color='#2a78d6',alpha=0.35,lw=0,label='50% interval')
        a.plot(g.target,g['median'],color='#2a78d6',lw=2,label='Median forecast')
        a.plot(g.target,g.obs,color='#3d3d3a',lw=1.6,marker='o',ms=2.5,label='Observed admissions')
        a.set_title(f"{'COVID-19' if p=='covid' else 'Influenza'} — {st}",loc='left',fontsize=10,color='#1f1f1d')
        a.grid(axis='y',color='#e6e5e0',lw=0.8); a.tick_params(axis='x',labelsize=8)
        import matplotlib.dates as mdates
        a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10])); a.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        if j==0: a.set_ylabel('Weekly admissions')
h,l=ax[0,0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=4,frameon=False,fontsize=9,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=(0,0.05,1,1)); fig.savefig('results/fig5_probabilistic.png',dpi=200,bbox_inches='tight')
