"""Figure 2: random inputs -> trained CNN -> Bernoulli samples, four sizes.

Run from project root using the TensorFlow environment:
  python ml/figure2_multisize.py
Outputs are isolated in a new timestamped directory. No training is performed.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import tensorflow as tf
from predict_any_size import load_predictor

PATTERNS = ['000', '001', '100', '101', '010', '011', '110', '111']
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
MARKERS = ['o', 's', '^', 'D']


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def wilson(k, n):
    if not n:
        return float('nan'), float('nan')
    z = 1.959963984540054
    phat = k/n
    center = (phat+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0.,center-half), min(1.,center+half)


def measure(model, length, p, samples, batch_size, seed_sequence):
    # Separate independent streams for initial configurations and sampling.
    initial_seed, sample_seed = seed_sequence.spawn(2)
    initial_rng = np.random.default_rng(initial_seed)
    sample_rng = np.random.default_rng(sample_seed)
    counts = np.zeros(8, np.int64)
    active = np.zeros(8, np.int64)
    sums = np.zeros(8, np.float64)
    abs_sums = np.zeros(8, np.float64)
    max_error = np.zeros(8, np.float64)
    theories = np.array([1-(1-p)**((code>>2)+(code&1)) for code in range(8)])
    for start in range(0, samples, batch_size):
        n = min(batch_size, samples-start)
        initial = initial_rng.integers(0, 2, (n,length), dtype=np.uint8)
        x = np.stack([initial.astype(np.float32), np.full((n,length),p,dtype=np.float32)],axis=-1)
        # Direct full-network evaluation, not a local lookup approximation.
        q = model(x, training=False).numpy()[...,0]
        sampled = sample_rng.random(q.shape) < q
        codes = (4*np.roll(initial,1,axis=1)+2*initial+np.roll(initial,-1,axis=1)).ravel()
        flat = q.ravel().astype(np.float64)
        errors = abs(flat-theories[codes])
        counts += np.bincount(codes,minlength=8)
        active += np.bincount(codes,weights=sampled.ravel(),minlength=8).astype(np.int64)
        sums += np.bincount(codes,weights=flat,minlength=8)
        abs_sums += np.bincount(codes,weights=errors,minlength=8)
        np.maximum.at(max_error,codes,errors)
    assert counts.sum() == samples*length
    assert np.all(active<=counts)
    rows=[]
    for pattern in PATTERNS:
        code=int(pattern,2); n=int(counts[code]); k=int(active[code])
        theory=float(theories[code])
        frequency=k/n if n else float('nan')
        predicted=float(sums[code]/n) if n else float('nan')
        low,high=wilson(k,n)
        rows.append(dict(structure=pattern,length=length,p=float(p),configurations=samples,
                         occurrences=n,sampled_active=k,theory=theory,
                         sampled_frequency=frequency,sampled_signed_error=frequency-theory,
                         sampled_absolute_error=abs(frequency-theory),
                         cnn_mean_probability=predicted,cnn_signed_error=predicted-theory,
                         cnn_absolute_error=abs(predicted-theory),
                         cnn_site_mae=float(abs_sums[code]/n) if n else float('nan'),
                         cnn_site_max_error=float(max_error[code]) if n else float('nan'),
                         sampling_minus_cnn=frequency-predicted,
                         sampled_wilson95_low=low,sampled_wilson95_high=high))
    return rows


def draw_panel(ax, rows, pattern, sizes, letter=None):
    ps=np.linspace(0,1,501)
    neighbors=int(pattern[0])+int(pattern[2])
    ax.plot(ps,1-(1-ps)**neighbors,color='black',lw=1.05,zorder=1)
    for series_index,(length,color,marker) in enumerate(zip(sizes,COLORS,MARKERS)):
        group=sorted([r for r in rows if r['structure']==pattern and r['length']==length],key=lambda r:r['p'])
        ax.plot([r['p'] for r in group],[r['sampled_frequency'] for r in group],
                linestyle='none',marker=marker,markersize=3.3,markerfacecolor='none',
                markeredgecolor=color,markeredgewidth=.65,alpha=.9,zorder=2,
                markevery=(series_index,4))
    title = f'[{pattern[0]}, {pattern[1]}, {pattern[2]}]'
    ax.text(.04,.96,title,transform=ax.transAxes,ha='left',va='top',fontsize=9)
    ax.legend(handles=legends(sizes),loc='lower right',bbox_to_anchor=(0.98, 0.1),fontsize=6,
              frameon=True,facecolor='white',framealpha=.9,edgecolor='none',
              borderpad=.35,labelspacing=.25,handlelength=1.5,handletextpad=.4)
    if letter:
        ax.text(-.16,1.08,letter,transform=ax.transAxes,fontsize=10,fontweight='bold')
    ax.set_xlim(-.025,1.025);ax.set_ylim(-.04,1.04)
    ax.set_xticks([0,.25,.5,.75,1]);ax.set_yticks([0,.25,.5,.75,1])
    ax.tick_params(direction='in',top=True,right=True,width=.6,length=3,labelsize=7)
    for spine in ax.spines.values():spine.set_linewidth(.65)


def legends(sizes):
    return [Line2D([],[],color='black',lw=1.05,label='Theory')]+[
        Line2D([],[],linestyle='none',marker=m,markersize=4,markerfacecolor='none',
               markeredgecolor=c,markeredgewidth=.8,label=f'L = {l}')
        for l,c,m in zip(sizes,COLORS,MARKERS)]


def save_figure(fig, path):
    for ext in ['pdf','svg','png']:
        fig.savefig(path.with_suffix('.'+ext),dpi=600,facecolor='white')
    plt.close(fig)


def main():
    base=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=base/'cnn_2_results'/'best_model.h5')
    parser.add_argument('--sizes',type=int,nargs=4,default=[32,64,128,256])
    parser.add_argument('--samples',type=int,default=2000,help='Random configurations per probability and size')
    parser.add_argument('--p-count',type=int,default=51)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--seed',type=int,default=20260921)
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    if len(set(args.sizes))!=4 or min(args.sizes)<3 or args.samples<1 or args.p_count<2 or args.batch_size<1:
        parser.error('Use four distinct lengths >=3, positive samples/batch size and p-count>=2')
    out=args.output or base/('figure2_multisize_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    # Never replace results from a previous experiment.
    out.mkdir(parents=True,exist_ok=False)
    (out/'individual').mkdir();(out/'error_data').mkdir()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    model=load_predictor(args.model)
    ps=np.linspace(0,1,args.p_count)
    roots=np.random.SeedSequence(args.seed).spawn(len(args.sizes)*len(ps))
    rows=[]
    for li,length in enumerate(args.sizes):
        for pi,p in enumerate(ps):
            rows.extend(measure(model,length,p,args.samples,args.batch_size,roots[li*len(ps)+pi]))
        print(f'Completed L={length}: {args.samples*len(ps):,} configurations',flush=True)
        write_csv(out/'error_data'/'all_probability_errors.csv',rows)
    summaries=[]
    for pattern in PATTERNS:
        for length in args.sizes:
            group=[r for r in rows if r['structure']==pattern and r['length']==length]
            write_csv(out/'error_data'/f'structure_{pattern}_L{length}.csv',group)
            valid=[r for r in group if r['occurrences']>0]
            sampled=np.array([r['sampled_signed_error'] for r in valid])
            cnn=np.array([r['cnn_signed_error'] for r in valid])
            summaries.append(dict(structure=pattern,length=length,
                total_occurrences=sum(r['occurrences'] for r in group),probabilities_evaluated=len(valid),
                sampled_mae_equal_p=float(np.abs(sampled).mean()) if valid else float('nan'),
                sampled_rmse_equal_p=float(np.sqrt(np.mean(sampled**2))) if valid else float('nan'),
                sampled_max_absolute_error=float(np.abs(sampled).max()) if valid else float('nan'),
                cnn_mae_equal_p=float(np.abs(cnn).mean()) if valid else float('nan'),
                cnn_max_absolute_error=float(np.abs(cnn).max()) if valid else float('nan')))
    write_csv(out/'error_data'/'structure_size_summary.csv',summaries)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'mathtext.fontset':'dejavusans',
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,4,figsize=(7.2,4.3),sharex=True,sharey=True)
    for ax,pattern,letter in zip(axes.flat,PATTERNS,'abcdefgh'):
        draw_panel(ax,rows,pattern,args.sizes,letter)
    fig.supxlabel(r'Transition probability $p$',fontsize=9,y=.035)
    fig.supylabel(r'Center-site activation frequency at $t+1$',fontsize=9,x=.015)
    fig.subplots_adjust(left=.10,right=.985,bottom=.15,top=.92,wspace=.16,hspace=.30)
    save_figure(fig,out/'figure2_all_structures')
    for pattern in PATTERNS:
        fig,ax=plt.subplots(figsize=(3.5,3.0))
        draw_panel(ax,rows,pattern,args.sizes)
        ax.set_xlabel(r'$p$',fontsize=9)
        ax.set_ylabel(r'$q$',fontsize=9)
        fig.subplots_adjust(left=.18,right=.97,bottom=.18,top=.94)
        save_figure(fig,out/'individual'/f'structure_{pattern}')
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    settings.update(output=str(out.resolve()),model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                    tensorflow_version=tf.__version__,numpy_version=np.__version__,
                    total_configurations=len(args.sizes)*len(ps)*args.samples,
                    total_site_occurrences=sum(args.sizes)*len(ps)*args.samples,
                    initial_distribution='Independent Bernoulli(0.5) at every site; new inputs for each p and L',
                    plotted_quantity='Frequency AFTER Bernoulli sampling of direct CNN probabilities',
                    boundary='periodic',summary_weighting='Equal weight per probability, not per occurrence',
                    intervals='Pointwise Wilson intervals for sampled frequency; not simultaneous intervals',
                    numerical_mode='TF32 disabled; dynamic CNN loaded from cnn_2 weights; no retraining')
    (out/'settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    caption=f'''Figure 2. Sampled one-step predictions across lattice sizes. For each of {len(ps)} probabilities in [0,1] and each lattice length L, {args.samples:,} independent random binary initial configurations are generated with site occupation probability 1/2. The CNN trained at L=32 is evaluated directly using the same weights and periodic boundaries; its output probabilities are then independently Bernoulli-sampled to obtain binary next-step configurations. Colored markers show the frequency of an active next-step center site, conditioned on the input triplet, pooling all lattice positions. Black solid curves show 1-(1-p)^(sigma_left+sigma_right). The eight panels are ordered as 000, 001, 100, 101, 010, 011, 110, 111. For readability, each size displays every fourth probability point with a different starting index, at its true p coordinate without artificial offsets; all probabilities are retained in the error tables. These are sampled frequencies, not the mean network probabilities. Statistical intervals and both sampling and network probability errors are supplied in the data files.\n'''
    (out/'figure_caption.txt').write_text(caption,encoding='utf-8')
    (out/'README.txt').write_text('individual/: eight separate panels in PDF/SVG/600 dpi PNG.\nfigure2_all_structures.*: combined eight-panel figure.\nerror_data/all_probability_errors.csv: counts, sampled frequencies, theory, signed/absolute errors, raw CNN probability errors and Wilson intervals.\nerror_data/structure_size_summary.csv: equal-p MAE/RMSE/max error per structure and size.\nerror_data/structure_XXX_LNN.csv: 32 separate structure/size tables.\nsettings.json: configuration, independent RNG seed streams, model hash and library versions.\nInitial states and sampled arrays are generated in batches and not retained; regenerate using this script and settings. Blue/orange/green/purple markers identify sizes; black lines are theory.\n',encoding='utf-8')
    assert sum(r['occurrences'] for r in rows)==settings['total_site_occurrences']
    print(f'All outputs saved to {out.resolve()}',flush=True)


if __name__=='__main__':
    main()
