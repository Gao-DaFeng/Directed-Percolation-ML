"""Figure 4: time-averaged density versus p, CNN (ML) versus direct MC.

Each realization starts fully occupied. Its density is averaged over every
integer t=0,...,T (including the initial state), then ensemble means and sample
standard deviations are computed across independent realizations. No training.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import tensorflow as tf

try:
    from .figure3_density import rollout_cnn, rollout_mc, write_csv
    from .predict_any_size import load_predictor
except ImportError:
    from figure3_density import rollout_cnn, rollout_mc, write_csv
    from predict_any_size import load_predictor


def time_average_statistics(counts, length):
    """Average time within each run; compute dispersion across whole runs."""
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[0] < 2 or counts.shape[1] < 2:
        raise ValueError("Expected counts[run,time] with >=2 runs and >=2 times")
    per_run = counts.mean(axis=1, dtype=np.float64) / length
    mean = float(per_run.mean())
    std = float(per_run.std(ddof=1))
    return per_run, mean, std, std / np.sqrt(len(per_run))


def summarize(out, ps, length, cnn_counts, mc_counts):
    rows, cnn_averages, mc_averages = [], [], []
    for p, cnn, mc in zip(ps, cnn_counts, mc_counts):
        cnn_avg, cnn_mean, cnn_std, cnn_sem = time_average_statistics(cnn, length)
        mc_avg, mc_mean, mc_std, mc_sem = time_average_statistics(mc, length)
        cnn_averages.append(cnn_avg); mc_averages.append(mc_avg)
        rows.append(dict(p=float(p), ml_mean=cnn_mean, mc_mean=mc_mean,
                         ml_std=cnn_std, mc_std=mc_std,
                         ml_standard_error=cnn_sem, mc_standard_error=mc_sem,
                         ml_minus_mc=cnn_mean-mc_mean,
                         absolute_difference=abs(cnn_mean-mc_mean),
                         difference_standard_error=float(np.hypot(cnn_sem, mc_sem)),
                         ml_mean_minus_std=cnn_mean-cnn_std, ml_mean_plus_std=cnn_mean+cnn_std,
                         mc_mean_minus_std=mc_mean-mc_std, mc_mean_plus_std=mc_mean+mc_std))
    write_csv(out / "density_vs_probability.csv", rows)
    np.savez_compressed(out / "time_averages_by_run.npz", probabilities=ps,
                        ml_time_averages=np.asarray(cnn_averages),
                        mc_time_averages=np.asarray(mc_averages))
    differences = np.array([row["ml_minus_mc"] for row in rows])
    largest = int(np.argmax(np.abs(differences)))
    errors = dict(mae_equal_weight_per_probability=float(abs(differences).mean()),
                  rmse_equal_weight_per_probability=float(np.sqrt(np.mean(differences**2))),
                  max_absolute_difference=float(abs(differences[largest])),
                  max_absolute_difference_p=float(ps[largest]),
                  interpretation="Differences include both model discrepancy and independent sampling noise")
    (out / "error_summary.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    return rows, errors


def plot_figure(out, rows, settings):
    ps = np.array([r["p"] for r in rows])
    mc = np.array([r["mc_mean"] for r in rows])
    ml = np.array([r["ml_mean"] for r in rows])
    mc_std = np.array([r["mc_std"] for r in rows])
    ml_std = np.array([r["ml_std"] for r in rows])
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "font.size": 11, "mathtext.fontset": "stix", "axes.linewidth": .8,
                         "text.color": "#252A31", "axes.labelcolor": "#252A31",
                         "xtick.color": "#252A31", "ytick.color": "#252A31",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    mc_color, ml_color = "#2463A6", "#D24B40"
    mc_band = ax.fill_between(ps, mc-mc_std, mc+mc_std, color=mc_color, alpha=.16,
                             linewidth=0, label=r"MC $\pm\,1\sigma$", zorder=1)
    ml_band = ax.fill_between(ps, ml-ml_std, ml+ml_std, color=ml_color, alpha=.16,
                             linewidth=0, label=r"ML $\pm\,1\sigma$", zorder=1)
    mc_line, = ax.plot(ps, mc, color=mc_color, lw=1.8, linestyle="-", label="MC mean", zorder=2)
    ml_line, = ax.plot(ps, ml, color=ml_color, lw=1.8, linestyle=(0, (4, 2.5)), label="ML mean", zorder=3)
    lower = min(-.025, float(np.min(np.r_[mc-mc_std, ml-ml_std]))-.015)
    upper = max(1.025, float(np.max(np.r_[mc+mc_std, ml+ml_std]))+.015)
    ax.set(xlabel=r"$p$", ylabel=r"$\rho_a$", xlim=(-.025, 1.025), ylim=(lower, upper))
    ax.set_xlabel(r"$p$", fontsize=15, labelpad=5)
    ax.set_ylabel(r"$\rho_a$", fontsize=16, labelpad=8)
    ax.set_xticks(np.linspace(0, 1, 6)); ax.set_yticks(np.linspace(0, 1, 6))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis="y", which="major", color="#E5E7EB", linewidth=.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=4, width=.8, pad=5, labelsize=10.5)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.2, width=.6)
    for spine in ax.spines.values():
        spine.set_color("#424850")
        spine.set_linewidth(.8)
    # Two rows pair each method's mean line with its standard-deviation band.
    ax.legend(handles=[mc_line, ml_line, mc_band, ml_band], loc="upper left",
              bbox_to_anchor=(.025, .98), fontsize=10, frameon=False, ncol=2,
              handlelength=2.5, handletextpad=.7, columnspacing=1.5,
              labelspacing=.65, borderaxespad=.4)
    # ax.text(.965, .085, fr"$L={settings['length']},\quad T={settings['steps']}$" + "\n"
    #         + fr"$N_{{\mathrm{{run}}}}={settings['runs_per_method_per_probability']}$",
    #         transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
    #         color="#58616B", linespacing=1.6)
    fig.subplots_adjust(left=.135, right=.975, bottom=.165, top=.965)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(out / ("figure4_density_vs_probability."+suffix), dpi=600, facecolor="white")
    plt.close(fig)


def write_documentation(out, settings):
    length = settings["length"]; steps = settings["steps"]
    runs = settings["runs_per_method_per_probability"]
    caption = (f"Figure 4. Probability dependence of the finite-time averaged particle density "
               f"on a periodic lattice of length L={length}, initialized in the fully occupied state. "
               f"For each p, MC and the trained CNN (ML) independently generate {runs:,} realizations "
               f"over t=0,...,{steps}. For realization r, rho_a,r = sum_(t=0)^{steps} "
               f"N_r(t)/[{length}*{steps+1}]. Blue solid and red dashed curves show the ensemble "
               f"means of rho_a,r for MC and ML, respectively. Shaded regions are the corresponding "
               f"means plus/minus one sample standard deviation across independent realizations "
               f"(ddof=1), not standard errors or confidence intervals. The t=0 initial state is "
               f"included, and extinct realizations contribute zero at subsequent times. The CNN "
               f"outputs are independently Bernoulli-sampled at each step and the sampled states "
               f"are fed back. MC uses independent left/right outgoing Bernoulli bonds and "
               f"synchronous updates. Lines join simulated p values without smoothing. "
               f"This is a finite-time average including the initial transient, not a stationary "
               f"density or a measurement restricted to t={steps}.\n")
    (out / "figure_caption.txt").write_text(caption, encoding="utf-8")
    (out / "README.txt").write_text(
        "figure4_density_vs_probability.pdf/.svg/.png: vector and 600 dpi figures.\n"
        "density_vs_probability.csv: means, sample SDs, standard errors, differences and band endpoints.\n"
        "density_runs.npz: cnn_counts/mc_counts axes [p, independent run, integer time], with full initial state.\n"
        "time_averages_by_run.npz: ML/MC time-averaged density for each independent realization.\n"
        "error_summary.json: differences between the two sample mean curves, with equal weight per p.\n"
        "settings.json: physical parameters, averaging definition, seeds, checkpoint and source hashes.\n"
        "The shaded SD describes spread across time-averaged runs; it is neither temporal SD nor SEM.\n"
        "ML is the pretrained CNN, evaluated directly at each step with Bernoulli-sampled state feedback.\n",
        encoding="utf-8")


def replot(source):
    settings = json.loads((source / "settings.json").read_text(encoding="utf-8"))
    with np.load(source / "density_runs.npz", allow_pickle=False) as data:
        ps = data["probabilities"]
        rows, errors = summarize(source, ps, int(data["length"]), data["cnn_counts"], data["mc_counts"])
    plot_figure(source, rows, settings)
    write_documentation(source, settings)
    print(json.dumps(errors, indent=2), flush=True)
    print(f"Replotted: {source.resolve()}", flush=True)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=base / "cnn_2_results" / "best_model.h5")
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--p-count", type=int, default=51, help="Equally spaced p values from 0 to 1")
    parser.add_argument("--probabilities", type=float, nargs="+", help="Optional explicit p grid")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replot", type=Path, help="Reuse saved run counts; no new simulations")
    args = parser.parse_args()
    if args.replot:
        replot(args.replot)
        return
    if args.length < 3 or args.runs < 2 or args.steps < 1 or args.p_count < 2 or args.batch_size < 1 or args.seed < 0:
        parser.error("Require length>=3, runs/p-count>=2, steps/batch-size>=1, seed>=0")
    ps = np.array(sorted(args.probabilities) if args.probabilities is not None else np.linspace(0, 1, args.p_count))
    if len(ps) < 2 or len(np.unique(ps)) != len(ps) or not np.all(np.isfinite(ps) & (ps >= 0) & (ps <= 1)):
        parser.error("At least two distinct finite probabilities in [0,1] are required")
    if not args.model.is_file():
        parser.error(f"Missing trained checkpoint: {args.model}")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    model = load_predictor(args.model)
    out = args.output or base.parent / "output" / "pdf" / (
        "figure4_density_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    # A figure-specific seed namespace keeps these draws separate from Figure 3.
    roots = np.random.SeedSequence(args.seed, spawn_key=(4,)).spawn(2 * len(ps))
    settings = dict(length=args.length, steps=args.steps, runs_per_method_per_probability=args.runs,
                    probabilities=ps.tolist(), total_realizations=int(2*len(ps)*args.runs),
                    initial_state="all occupied", boundary="periodic", times_included=list(range(args.steps+1)),
                    observable="Each run: mean over t=0,...,T of N(t)/L; then mean across runs",
                    band="Mean +/- one sample standard deviation of per-run time averages, ddof=1",
                    standard_error="Sample SD of per-run time averages divided by sqrt(runs)",
                    includes_initial_state=True, includes_extinct_runs=True,
                    model=str(args.model.resolve()), model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                    source_hashes={name:hashlib.sha256((base/name).read_bytes()).hexdigest()
                                   for name in ("figure4_density_vs_probability.py", "figure3_density.py", "predict_any_size.py")},
                    seed=args.seed, seed_namespace=[4], batch_size=args.batch_size,
                    random_streams=[dict(p=float(p), ml=list(roots[2*i].spawn_key), mc=list(roots[2*i+1].spawn_key))
                                    for i,p in enumerate(ps)],
                    tensorflow_version=tf.__version__, numpy_version=np.__version__, tf32_enabled=False,
                    training_performed=False, theory_correction=False,
                    mc="Two independent outgoing bonds per source; synchronous periodic update",
                    cnn="Direct CNN inference each step, independent Bernoulli sampling, binary-state feedback")
    (out / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"Output: {out.resolve()}", flush=True)
    start = time.perf_counter()
    cnn_counts, mc_counts = [], []
    for i, p in enumerate(ps):
        cnn, _ = rollout_cnn(model, args.length, args.runs, args.steps, p,
                             np.random.default_rng(roots[2*i]), args.batch_size)
        mc, _ = rollout_mc(args.length, args.runs, args.steps, p,
                          np.random.default_rng(roots[2*i+1]))
        cnn_counts.append(cnn); mc_counts.append(mc)
        if i % 5 == 0 or i == len(ps)-1:
            print(f"{i+1}/{len(ps)} p={p:.4f}: ML rho_a={cnn.mean()/args.length:.6f}, "
                  f"MC rho_a={mc.mean()/args.length:.6f}; elapsed {time.perf_counter()-start:.1f}s", flush=True)
    cnn_counts, mc_counts = np.asarray(cnn_counts), np.asarray(mc_counts)
    np.savez_compressed(out / "density_runs.npz", probabilities=ps, times=np.arange(args.steps+1),
                        length=args.length, runs=args.runs, cnn_counts=cnn_counts, mc_counts=mc_counts)
    rows, errors = summarize(out, ps, args.length, cnn_counts, mc_counts)
    plot_figure(out, rows, settings)
    write_documentation(out, settings)
    print(json.dumps(errors, indent=2), flush=True)
    print(f"Completed: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
