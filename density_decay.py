"""Full-occupancy density decay near the CNN critical point (reference panel a)."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from critical_dynamics import network_table, theory_table, simulate, fit_decay, write_csv
from predict_any_size import load_predictor


# 绘制临界指数图三个概率[.64,.643,.65,]，得到临界点和临界指数

def main():
    base=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probabilities',nargs='+',type=float,default=[.64,.643,.65,])
    p.add_argument('--length',type=int,default=10000)
    p.add_argument('--runs',type=int,default=1000)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--fit-start',type=int,default=10)
    p.add_argument('--fit-stop',type=int,default=1000)
    p.add_argument('--seed',type=int,default=20260922)
    p.add_argument('--engine',choices=['cnn','theory'],default='cnn')
    p.add_argument('--output',type=Path,default=base/'critical_panel_a_scan')
    args=p.parse_args()
    if args.length<3 or args.runs<2 or not 1<=args.fit_start<args.fit_stop<=args.steps:
        p.error('Invalid dimensions or fitting window')
    if len(args.probabilities)<3 or any(not 0<=v<=1 for v in args.probabilities):p.error('Supply at least three probabilities in [0,1]')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    model=load_predictor(base/'cnn_2_results/best_model.h5') if args.engine=='cnn' else None
    times=np.unique(np.r_[0,np.geomspace(1,args.steps,320).astype(int)])
    rows=[];curves=[];raw=[]
    for i,prob in enumerate(sorted(set(args.probabilities))):
        table=network_table(model,prob) if model is not None else theory_table(prob)
        density,deaths=simulate(table,args.length,args.runs,args.steps,times,args.seed+i*100003)
        mean=density.mean(0);sem=density.std(0,ddof=1)/np.sqrt(args.runs)
        np.savez_compressed(out/f'{args.engine}_p{prob:.7f}.npz',p=prob,times=times,density_by_run=density,first_zero=deaths,local_table=table)
        fit=fit_decay(times,mean,args.fit_start,args.fit_stop)
        if fit:rows.append(dict(p=prob,**fit,extinct_fraction=float(np.mean(deaths<=args.steps))))
        curves.append((prob,mean,sem))
        for t,m,se in zip(times,mean,sem):raw.append(dict(p=prob,t=int(t),density=float(m),standard_error=float(se)))
        print(f'p={prob:.7f}: {fit}; rho(T)={mean[-1]:.5f}',flush=True)
    write_csv(out/'density.csv',raw);write_csv(out/'fits.csv',rows)
    brackets=[(a,b) for a,b in zip(rows[:-1],rows[1:]) if a['curvature']<0<b['curvature']]
    if brackets:
        a,b=min(brackets,key=lambda pair:abs(pair[0]['curvature'])+abs(pair[1]['curvature']))
        candidate=min((a,b),key=lambda r:abs(r['curvature']));bracket=[a['p'],b['p']]
    else:
        candidate=min(rows,key=lambda r:abs(r['curvature'])) if rows else None;bracket=None
    result=dict(candidate=candidate,curvature_bracket=bracket,interpretation='Finite-time candidate only. Confirm with narrower p spacing, fit-window and size checks.',
                settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    (out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'mathtext.fontset':'dejavusans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(4.4,3.6))
    colors=plt.cm.viridis(np.linspace(.1,.9,len(curves)))
    for (prob,mean,sem),color in zip(curves,colors):
        ax.loglog(times[1:],np.where(mean[1:]>0,mean[1:],np.nan),color=color,lw=1.1,label=f'p = {prob:.5f}')
        ax.fill_between(times[1:],np.maximum(mean[1:]-1.96*sem[1:],1e-8),mean[1:]+1.96*sem[1:],color=color,alpha=.10,lw=0)
    if candidate:
        prob,mean,_=next(c for c in curves if c[0]==candidate['p'])
        mask=(times>=args.fit_start)&(times<=args.fit_stop)&(mean>0)
        slope,intercept=np.polyfit(np.log(times[mask]),np.log(mean[mask]),1)
        tt=np.geomspace(args.fit_start,args.fit_stop,150)
        ax.loglog(tt,np.exp(intercept)*tt**slope,'--',color='red',lw=1.3,label=fr'Fit: $\delta={-slope:.4f}$')
    ax.set(xlabel=r'$t$',ylabel=r'$\rho(t)$',xlim=(1,args.steps))
    ax.tick_params(which='both',direction='in',top=True,right=True)
    ax.legend(frameon=False,fontsize=7,loc='lower left')
    # ax.text(.97,.96,f'{args.engine.upper()}, L={args.length}\n{args.runs} independent runs',transform=ax.transAxes,ha='right',va='top',fontsize=8)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(out/f'density_decay.{ext}',dpi=600)
    plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
