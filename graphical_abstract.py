import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager
fam='Inter' if any('Inter' in f.name for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams['font.family']=fam
W,H=13.28,5.31
fig=plt.figure(figsize=(W,H),dpi=200)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis('off')
ink='#1f1f1f'; mut='#5c5c5c'
ww='#2a78d6'; hosp='#1baf7a'; pure='#eb6834'; const='#6250d6'
def box(x,y,w,h,ec,title):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.18',fc='white',ec=ec,lw=2.2))
    ax.text(x+w/2,y+h-0.38,title,ha='center',va='center',fontsize=15,weight='bold',color=ink)
def arrow(x1,x2,y):
    ax.add_patch(FancyArrowPatch((x1,y),(x2,y),arrowstyle='-|>',mutation_scale=26,lw=2.4,color=mut))
ax.text(W/2,H-0.38,'Hybrid SIR-IVP + machine learning: forecasting hospital admissions and allocating surge capacity',ha='center',va='center',fontsize=16.5,weight='bold',color=ink)
y0,h=0.35,4.15
# panel 1 data
box(0.3,y0,3.6,h,hosp,'Data (10 U.S. states)')
t=np.linspace(0,6*np.pi,200)
a1=fig.add_axes([0.3/W+0.02,(y0+1.55)/H,3.6/W-0.04,1.75/H])
hh=np.exp(np.sin(t/1.6)+0.3*np.sin(t*0.9)); wwv=np.exp(np.sin((t+1.2)/1.6)+0.3*np.sin((t+1.2)*0.9))
a1.plot(t,hh,color=hosp,lw=2.4); a1.plot(t,wwv*0.9,color=ww,lw=2.4,ls='--'); a1.axis('off')
ax.text(2.1,y0+1.25,'Weekly hospital admissions',ha='center',fontsize=12,color=hosp,weight='bold')
ax.text(2.1,y0+0.82,'Wastewater (NWSS)',ha='center',fontsize=12,color=ww,weight='bold')
ax.text(2.1,y0+0.38,'COVID-19 and influenza, 2022–2026',ha='center',fontsize=11,color=mut)
arrow(3.95,4.6,y0+h/2)
# panel 2 model
box(4.65,y0,3.95,h,const,'Model with proofs')
items=[('1','Exact weekly reconstruction of β'),('','unique, scale-free, stable'),('2','Gradient boosting forecasts log β'),('','wastewater as optional input'),('3','Propagate through SIR-IVP'),('','error bound: rate errors accumulate'),('4','Allocate surge reserve'),('','predictive distributions (water-filling)')]
yy=y0+h-1.0
for n,s in items:
    if n:
        ax.text(4.95,yy,n,fontsize=12.5,weight='bold',color=const,va='center')
        ax.text(5.25,yy,s,fontsize=12.5,color=ink,va='center')
        yy-=0.36
    else:
        ax.text(5.25,yy,s,fontsize=11,color=mut,va='center',style='italic'); yy-=0.48
arrow(8.65,9.3,y0+h/2)
# panel 3 results
box(9.35,y0,3.63,h,ww,'Key results')
res=[('24–41%','lower WIS than persistence\n(2–4 weeks ahead)'),('≈ CDC','ensemble WIS on hospital\ndata as first published'),('½ error','during hospital-reporting\noutages, using wastewater'),('74→92%','of attainable benefit with\ndistributions (20% reserve)')]
yy=y0+h-1.05
for big,small in res:
    ax.text(9.55,yy,big,fontsize=15,weight='bold',color=ww,va='center')
    ax.text(10.95,yy,small,fontsize=10,color=ink,va='center',linespacing=1.15)
    yy-=0.82
fig.savefig('graphical_abstract.png',dpi=200)
from PIL import Image; im=Image.open('graphical_abstract.png'); print(im.size)
