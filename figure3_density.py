"""Figure 3: independent CNN and MC density rollouts from full occupation.

Defaults: L=32, t=0,...,50, 2000 runs per method per p, periodic boundaries.
The trained cnn_2 checkpoint is evaluated directly at every step. No training,
probability lookup, teacher forcing, or theoretical correction is performed.
"""
import argparse
import csv
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
from matplotlib.lines import Line2D
import numpy as np
import tensorflow as tf

try:
    from .predict_any_size import load_predictor
except ImportError:
    from predict_any_size import load_predictor


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#A6761D"]
MARKERS = ["o", "s", "^", "D", "v"]


def mc_step(states, p, rng):
    """Two independent outgoing bonds per source; simultaneous periodic update."""
    left_bonds = states & (rng.random(states.shape) < p)
    right_bonds = states & (rng.random(states.shape) < p)
    return (np.roll(left_bonds, -1, axis=1)
            | np.roll(right_bonds, 1, axis=1)).astype(np.uint8)


def rollout_mc(length, runs, steps, p, rng):
    states = np.ones((runs, length), dtype=np.uint8)
    counts = np.empty((runs, steps + 1), dtype=np.int32)
    counts[:, 0] = length
    for t in range(1, steps + 1):
        states = mc_step(states, p, rng)
        counts[:, t] = states.sum(axis=1)
    return counts, states


def rollout_cnn(model, length, runs, steps, p, rng, batch_size):
    states = np.ones((runs, length), dtype=np.uint8)
    counts = np.empty((runs, steps + 1), dtype=np.int32)
    counts[:, 0] = length
    inputs = np.empty((min(batch_size, runs), length, 2), dtype=np.float32)
    inputs[..., 1] = p
    for t in range(1, steps + 1):
        # Every batch reads the complete previous-time state; commit after all
        # sites/runs have been updated to prevent asynchronous site updates.
        following = np.empty_like(states)
        for start in range(0, runs, batch_size):
            stop = min(start + batch_size, runs)
            n = stop - start
            inputs[:n, :, 0] = states[start:stop]
            q = model(inputs[:n], training=False).numpy()[..., 0]
            if q.shape != (n, length) or not np.all(np.isfinite(q) & (q >= 0) & (q <= 1)):
                raise ValueError("CNN must output one finite probability in [0,1] per site")
            following[start:stop] = rng.random(q.shape) < q
        states = following
        counts[:, t] = states.sum(axis=1)
    return counts, states


def density_statistics(counts, length):
    densities = counts.astype(np.float64) / length
    return densities.mean(axis=0), densities.std(axis=0, ddof=1) / np.sqrt(len(counts))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(out, probabilities, times, cnn_counts, mc_counts, length):
    rows, errors, curves = [], [], []
    for p, cnn, mc in zip(probabilities, cnn_counts, mc_counts):
        cnn_mean, cnn_se = density_statistics(cnn, length)
        mc_mean, mc_se = density_statistics(mc, length)
        difference = cnn_mean - mc_mean
        difference_se = np.hypot(cnn_se, mc_se)
        for i, t in enumerate(times):
            rows.append(dict(p=float(p), t=int(t), cnn_density=cnn_mean[i],
                             mc_density=mc_mean[i], cnn_standard_error=cnn_se[i],
                             mc_standard_error=mc_se[i], cnn_minus_mc=difference[i],
                             difference_standard_error=float(difference_se[i]),
                             difference_ci95_low=float(difference[i] - 1.96*difference_se[i]),
                             difference_ci95_high=float(difference[i] + 1.96*difference_se[i]),
                             cnn_zero_fraction=float(np.mean(cnn[:, i] == 0)),
                             mc_zero_fraction=float(np.mean(mc[:, i] == 0))))
        # Exclude the identically occupied initial state from curve-error metrics.
        errors.append(dict(p=float(p), time_start=1, time_stop=int(times[-1]),
                           density_mae=float(np.abs(difference[1:]).mean()),
                           density_rmse=float(np.sqrt(np.mean(difference[1:]**2))),
                           max_absolute_error=float(np.abs(difference[1:]).max()),
                           max_absolute_error_time=int(times[1:][np.argmax(np.abs(difference[1:]))]),
                           cnn_final_density=float(cnn_mean[-1]),
                           mc_final_density=float(mc_mean[-1])))
        curves.append((cnn_mean, mc_mean, difference_se))
    write_csv(out / "density_statistics.csv", rows)
    write_csv(out / "density_errors.csv", errors)
    return curves, errors


def plot_figure(out, probabilities, times, curves, length, runs, marker_every):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "mathtext.fontset": "dejavusans", "axes.linewidth": .8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, (ax, residual) = plt.subplots(2, 1, sharex=True, figsize=(5.5, 5.9),
                                      gridspec_kw={"height_ratios": [1.25, 1], "hspace": .13})
    ml_handles, mc_handles, residual_handles = [], [], []
    residual_extent = 0.
    for i, (p, (cnn, mc, difference_se)) in enumerate(zip(probabilities, curves)):
        color, marker = COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]
        ax.plot(times, cnn, color=color, linestyle="-", lw=1.05, marker="o", ms=3.8,
                markerfacecolor="none", markeredgecolor=color, markeredgewidth=.85,
                markevery=marker_every, zorder=2, clip_on=False)
        ax.plot(times, mc, color=color, linestyle="--", lw=1.05, marker="s", ms=3.8,
                markerfacecolor="none", markeredgecolor=color, markeredgewidth=.85,
                markevery=(marker_every//2, marker_every), zorder=3, clip_on=False)
        ml_handles.append(Line2D([], [], color=color, linestyle="-", marker="o", ms=4.2,
                                 markerfacecolor="none", label=fr"ML, $p={p:g}$"))
        mc_handles.append(Line2D([], [], color=color, linestyle="--", marker="s", ms=4.2,
                                 markerfacecolor="none", label=fr"MC, $p={p:g}$"))
        residual_handles.append(Line2D([], [], color=color, lw=1.05, marker=marker, ms=3.2,
                                       markerfacecolor="white", label=fr"$p={p:g}$"))
        # Display signed absolute density differences in explicit 10^-3 units.
        # Independent MC/CNN ensembles imply Var(diff) = SE_CNN^2 + SE_MC^2.
        difference = 1000. * (cnn - mc)
        half_width = 1000. * 1.96 * difference_se
        residual.fill_between(times, difference-half_width, difference+half_width,
                              color=color, alpha=.10, linewidth=0, zorder=1)
        residual.plot(times, difference, color=color, lw=1.05, marker=marker,
                      ms=3.2, markerfacecolor="white", markeredgewidth=.75,
                      markevery=max(1, marker_every*2), zorder=3)
        residual_extent = max(residual_extent, float(np.max(abs(difference) + half_width)))
    ax.set(ylabel=r"$\rho(t)$",
           xlim=(0, int(times[-1])), ylim=(-.025, 1.025))
    ax.set_yticks(np.linspace(0, 1, 6))
    if times[-1] == 50:
        ax.set_xticks(np.arange(0, 51, 10))
    for panel, letter in ((ax, "(a)"), (residual, "(b)")):
        panel.tick_params(which="both", direction="in", top=True, right=True, length=3.5)
        panel.text(.025, .92, letter, transform=panel.transAxes,
                   ha="left", va="top", fontweight="bold", fontsize=10)
    residual.axhline(0, color="0.25", lw=.9, linestyle="--", zorder=2)
    extent = max(1., residual_extent*1.10)
    residual.set(xlabel=r"$t$", ylabel=r"$\Delta\rho(t)$  $(\times 10^{-3})$",
                 ylim=(-extent, extent))
    # Matplotlib fills legend columns: each row pairs ML and MC at the same p.
    ax.legend(handles=ml_handles + mc_handles, loc="lower right", bbox_to_anchor=(.98, .12),
              frameon=False, fontsize=8, ncol=2, labelspacing=.45,
              handlelength=2.5, handletextpad=.4, columnspacing=1.1)
    residual.legend(handles=residual_handles, loc="lower center", bbox_to_anchor=(.5, .025),
                    frameon=False, fontsize=7.5, ncol=len(probabilities), labelspacing=.4,
                    handlelength=1.5, handletextpad=.35, columnspacing=.8)
    fig.subplots_adjust(left=.16, right=.97, bottom=.10, top=.97)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(out / ("figure3_density." + suffix), dpi=600, facecolor="white")
    plt.close(fig)


def write_caption(out, probabilities, times, length, runs, marker_every):
    caption = (f"Figure 3. Density evolution and CNN-MC differences from a fully occupied "
               f"initial configuration on a periodic lattice with L={length}. For each p "
               f"({', '.join(f'{p:g}' for p in probabilities)}), each method uses {runs:,} independent "
               f"realizations over t=0,...,{int(times[-1])}. (a) Ensemble-averaged densities: "
               f"solid lines with open circles show autoregressive CNN simulations (ML) using "
               f"the checkpoint trained by cnn_2.py; dashed lines with open squares show MC. "
               f"The same probability has the same color for both methods. All probabilities use "
               f"circles for ML and squares for MC. The in-panel legend pairs ML and MC at each p. "
               f"The CNN probabilities are independently Bernoulli-"
               f"sampled at every step and binary configurations are fed back. Symbols in (a) "
               f"are displayed every {marker_every} steps, with CNN starting at t=0 and MC at "
               f"t={marker_every//2} to reduce marker overlap; all curves retain every integer "
               f"time without coordinate offsets. (b) Signed density differences "
               f"Delta rho = rho_CNN - rho_MC in units of 10^-3; curves include every integer "
               f"time. The dashed line marks zero difference. Shaded bands are approximate "
               f"pointwise normal 95% confidence intervals, Delta rho +/- 1.96 sqrt(SE_CNN^2 "
               f"+ SE_MC^2), with each SE calculated across independent run densities. Colors "
               f"identify p in both panels; the legend in (b) identifies its colored difference "
               f"curves and marker shapes. MC and CNN use independent "
               f"random streams, and all runs including zero-density runs contribute to the "
               f"averages. These are not simultaneous confidence bands; time points are correlated. "
               f"Differences include model discrepancy and sampling fluctuations.\n")
    (out / "figure_caption.txt").write_text(caption, encoding="utf-8")
    (out / "plot_settings.json").write_text(json.dumps(dict(
        panels=["density", "signed_density_difference"], difference_display_scale=1000,
        interval="Pointwise normal approximation: difference +/- 1.96 * hypot(SE_CNN, SE_MC)",
        simultaneous_interval=False, markers_every=marker_every,
        cnn_linestyle="solid", mc_linestyle="dashed", shared_color_per_probability=True,
        density_markers_by_method=dict(ML="o", MC="s"),
        density_legend="ML, p=... and MC, p=... paired by probability; inside lower right",
        difference_legend="Probability labels; inside lower center",
        cnn_marker_start=0, mc_marker_start=marker_every//2,
        all_times_plotted_in_difference=True), indent=2), encoding="utf-8")


def replot_saved(source, out, marker_every):
    """Regenerate statistics/figures from stored counts, without loading CNN."""
    with np.load(source / "density_runs.npz", allow_pickle=False) as saved:
        probabilities, times = saved["probabilities"], saved["times"]
        length, runs = int(saved["length"]), int(saved["runs"])
        cnn_counts, mc_counts = saved["cnn_counts"], saved["mc_counts"]
    if cnn_counts.shape != (len(probabilities), runs, len(times)) or mc_counts.shape != cnn_counts.shape:
        raise ValueError("Saved count arrays do not match probability/run/time dimensions")
    if out.resolve() != source.resolve():
        out.mkdir(parents=True, exist_ok=False)
        (out / "source_data.txt").write_text(str((source / "density_runs.npz").resolve()), encoding="utf-8")
    curves, errors = summarize(out, probabilities, times, cnn_counts, mc_counts, length)
    plot_figure(out, probabilities, times, curves, length, runs, marker_every)
    write_caption(out, probabilities, times, length, runs, marker_every)
    print(json.dumps(errors, indent=2), flush=True)
    print(f"Replotted saved data: {out.resolve()}", flush=True)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=base / "cnn_2_results" / "best_model.h5")
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--runs", type=int, default=2000, help="Independent runs per method per probability")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--probabilities", type=float, nargs="+", default=[.3, .5, .644, .7, .9])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--marker-every", type=int, default=2, help="Marker display interval; all steps are saved")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replot", type=Path, help="Existing results folder; reuse saved counts without simulations")
    args = parser.parse_args()
    if args.length < 3 or args.runs < 2 or args.steps < 1 or args.batch_size < 1 or args.marker_every < 1 or args.seed < 0:
        parser.error("Require length>=3, runs>=2, steps/batch-size/marker-every>=1 and seed>=0")
    if args.replot:
        replot_saved(args.replot, args.output or args.replot, args.marker_every)
        return
    if (not np.all(np.isfinite(args.probabilities))
            or any(p < 0 or p > 1 for p in args.probabilities)
            or len(set(args.probabilities)) != len(args.probabilities)):
        parser.error("Probabilities must be distinct finite numbers in [0,1]")
    if not args.model.is_file():
        parser.error(f"Trained checkpoint is missing: {args.model}")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    model = load_predictor(args.model)
    out = args.output or base.parent / "output" / "pdf" / (
        "figure3_density_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    probabilities = np.asarray(args.probabilities, dtype=np.float64)
    times = np.arange(args.steps + 1, dtype=np.int64)
    seeds = np.random.SeedSequence(args.seed).spawn(2 * len(probabilities))
    settings = dict(length=args.length, runs_per_method_per_probability=args.runs,
                    steps=args.steps, probabilities=probabilities.tolist(), seed=args.seed,
                    model=str(args.model.resolve()), model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                    tensorflow_version=tf.__version__, numpy_version=np.__version__,
                    batch_size=args.batch_size, initial_state="all occupied", boundary="periodic",
                    cnn="Direct full CNN then independent Bernoulli sampling; sampled states fed back at each step",
                    mc="Two independent outgoing Bernoulli(p) bonds per active source, synchronous update",
                    averaging="Unconditional mean over all runs including zero-density runs",
                    rng="Separate PCG64 streams for each probability and method; no reseeding across time",
                    stream_spawn_keys=[dict(p=float(p), cnn=list(seeds[2*i].spawn_key),
                                            mc=list(seeds[2*i+1].spawn_key)) for i, p in enumerate(probabilities)],
                    markers_every=args.marker_every, standard_error="Across independent run densities, not sites",
                    training_performed=False, theory_correction=False, tf32_enabled=False)
    (out / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"Output: {out.resolve()}", flush=True)
    cnn_counts, mc_counts, cnn_final, mc_final = [], [], [], []
    for i, p in enumerate(probabilities):
        start = time.perf_counter()
        cnn, cnn_states = rollout_cnn(model, args.length, args.runs, args.steps, p,
                                      np.random.default_rng(seeds[2*i]), args.batch_size)
        mc, mc_states = rollout_mc(args.length, args.runs, args.steps, p,
                                   np.random.default_rng(seeds[2*i+1]))
        cnn_counts.append(cnn); mc_counts.append(mc)
        cnn_final.append(cnn_states); mc_final.append(mc_states)
        print(f"p={p:g}: CNN rho({args.steps})={cnn[:,-1].mean()/args.length:.6f}, "
              f"MC={mc[:,-1].mean()/args.length:.6f}; {time.perf_counter()-start:.1f}s", flush=True)
    cnn_counts, mc_counts = np.asarray(cnn_counts), np.asarray(mc_counts)
    np.savez_compressed(out / "density_runs.npz", probabilities=probabilities, times=times,
                        length=args.length, runs=args.runs, cnn_counts=cnn_counts, mc_counts=mc_counts,
                        cnn_final_states=np.asarray(cnn_final), mc_final_states=np.asarray(mc_final))
    curves, errors = summarize(out, probabilities, times, cnn_counts, mc_counts, args.length)
    plot_figure(out, probabilities, times, curves, args.length, args.runs, args.marker_every)
    write_caption(out, probabilities, times, args.length, args.runs, args.marker_every)
    (out / "README.txt").write_text(
        "figure3_density.pdf/.svg/.png: vector formats and 600 dpi PNG.\n"
        "Panel (a): mean densities; panel (b): signed CNN-MC differences in 10^-3 units.\n"
        "Shading: approximate pointwise normal 95% intervals, not simultaneous bands.\n"
        "plot_settings.json: display scale and confidence-interval definition.\n"
        "density_statistics.csv: all p/time means, standard errors and CNN-MC differences.\n"
        "density_errors.csv: curve MAE/RMSE/max error over t=1,...,T (excluding t=0).\n"
        "density_runs.npz: cnn_counts/mc_counts have axes [p, independent run, time]; divide by L for density.\n"
        "Initial configurations are all ones. Final sampled binary configurations are also saved.\n"
        "settings.json: parameters, independent RNG stream identifiers, checkpoint SHA256 and versions.\n"
        "No training or theoretical probability correction is applied. Zero-density CNN states are not manually clamped.\n"
        "Standard errors reflect sampling uncertainty, not network bias.\n", encoding="utf-8")
    print(json.dumps(errors, indent=2), flush=True)
    print(f"Completed: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
