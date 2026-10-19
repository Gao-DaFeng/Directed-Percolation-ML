from pathlib import Path
import csv
import json
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def prepare_single_step_dataset(initial_configs, probabilities, next_configs):
    """Convert one-step pairs into inputs (N, L, 2) and targets (N, L, 1)."""
    initial_configs = initial_configs.astype("float32")
    next_configs = next_configs.astype("float32")
    p_channel = np.broadcast_to(
        probabilities.astype("float32")[:, None], initial_configs.shape
    )
    X_input = np.stack([initial_configs, p_channel], axis=-1)
    y = next_configs[..., None]
    return X_input, y




def structure_sample_weights(initial_configs):
    """Equal total BCE weight per observed triplet; mean site weight is one."""
    states = initial_configs.astype(np.uint8)
    codes = 4 * np.roll(states, 1, axis=1) + 2 * states + np.roll(states, -1, axis=1)
    counts = np.bincount(codes.ravel(), minlength=8)
    present = counts > 0
    weights = np.zeros(8, dtype=np.float64)
    weights[present] = codes.size / (present.sum() * counts[present])
    return weights[codes].astype(np.float32), counts, weights


def compare_local_structures(model, X_test, y_test, output_dir, tolerance=0.02):
    """Compare center-site activation probabilities for all eight triplets.

    Scan every site (including the periodic seam). Agreement means absolute
    probability error <= tolerance, not equality of individual random outcomes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states = X_test[..., 0].astype(np.uint8)
    ps = X_test[:, 0, 1]
    actual = y_test[..., 0]
    predicted = model.predict(X_test, batch_size=256, verbose=0)[..., 0]
    # Binary code for [left, center, right].
    codes = 4 * np.roll(states, 1, axis=1) + 2 * states + np.roll(states, -1, axis=1)
    p_values = np.unique(ps)
    # Same panel order as the reference image.
    patterns = [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0),
                (0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]
    rows, summaries = [], []
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    for ax, pattern in zip(axes.flat, patterns):
        code = 4 * pattern[0] + 2 * pattern[1] + pattern[2]
        neighbor_count = pattern[0] + pattern[2]
        group_rows = []
        for p in p_values:
            sample_mask = ps == p
            site_mask = codes[sample_mask] == code
            n = int(site_mask.sum())
            theory = 1.0 - (1.0 - float(p)) ** neighbor_count
            pred = predicted[sample_mask][site_mask]
            observed = actual[sample_mask][site_mask]
            error = np.abs(pred - theory)
            row = {
                'structure': ''.join(map(str, pattern)), 'p': float(p), 'count': n,
                'predicted_mean': float(pred.mean()) if n else float('nan'),
                'observed_frequency': float(observed.mean()) if n else float('nan'),
                'theory': theory,
                'mean_probability_error': abs(float(pred.mean()) - theory) if n else float('nan'),
                'site_probability_mae': float(error.mean()) if n else float('nan'),
                'site_probability_max_error': float(error.max()) if n else float('nan'),
                'sites_within_tolerance': int((error <= tolerance).sum()),
                'tolerance': tolerance,
                'mean_within_tolerance': bool(abs(float(pred.mean()) - theory) <= tolerance) if n else '',
            }
            group_rows.append(row)
        rows.extend(group_rows)
        valid = [row for row in group_rows if row['count']]
        total = sum(row['count'] for row in valid)
        summary = {
            'structure': ''.join(map(str, pattern)), 'count': total,
            'probability_mae': sum(row['site_probability_mae'] * row['count'] for row in valid) / total if total else float('nan'),
            'max_probability_error': max((row['site_probability_max_error'] for row in valid), default=float('nan')),
            'fraction_within_tolerance': sum(row['sites_within_tolerance'] for row in valid) / total if total else float('nan'),
            'groups_within_tolerance': sum(row['mean_within_tolerance'] for row in valid),
            'groups_evaluated': len(valid), 'tolerance': tolerance,
        }
        summaries.append(summary)
        print(f"Structure {pattern}: N={total}, MAE={summary['probability_mae']:.6f}, "
              f"fraction within {tolerance}={summary['fraction_within_tolerance']:.4f}")
        for key, label, style in [('observed_frequency', 'Observed (all samples)', '-'),
                                  ('predicted_mean', 'ML (all samples)', '--'),
                                  ('theory', 'Theory', ':')]:
            ax.plot(p_values, [row[key] for row in group_rows], style, label=label)
        ax.set(title=f'init={list(pattern)} | N={total:,}', xlabel='p',
               ylabel='P(next center = 1)', ylim=(-0.03, 1.03))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(
        f'All input triplets (training = evaluation): {states.shape[0]:,} configurations x '
        f'{states.shape[1]} sites = {states.size:,} occurrences (periodic boundaries)',
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / 'structure_comparison.png', dpi=180)
    plt.close(fig)
    for filename, records in [('structure_by_probability.csv', rows),
                              ('structure_summary.csv', summaries)]:
        with (output_dir / filename).open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(f'Structure comparisons saved to {output_dir.resolve()}')


np.random.seed(42)
tf.random.set_seed(42)
output_dir = Path(__file__).resolve().parent / "cnn_2_results"
output_dir.mkdir(parents=True, exist_ok=True)

data_path = Path(__file__).resolve().parents[1] / "data_mc" / "data_32_2.npz"
with np.load(data_path, allow_pickle=False) as data:
    initial_configs = data["initial_configs"]
    probabilities = data["probabilities"]
    next_configs = data["next_configs"]

X_input, y = prepare_single_step_dataset(initial_configs, probabilities, next_configs)
# Use every sample for both fitting and evaluation (no held-out split).
train_indices = np.arange(len(X_input))
test_indices = train_indices.copy()
X_train, y_train = X_input, y
X_test, y_test = X_input, y
# BCE reduces the final singleton channel, so sample_weight has shape (N, L).
site_weights, structure_counts, structure_weights = structure_sample_weights(initial_configs)
with (output_dir / "structure_weights.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.writer(handle)
    writer.writerow(["structure", "count", "site_weight"])
    for code in range(8):
        writer.writerow([format(code, "03b"), int(structure_counts[code]), float(structure_weights[code])])
print("X_input shape:", X_input.shape)  # (N_samples, 32, 2)
print("y shape:", y.shape)              # (N_samples, 32, 1)

L = X_input.shape[1]  # 32
in_channels = X_input.shape[2]  # 2

inputs = keras.Input(shape=(L, in_channels))  # (L, 2)

# Wrap the last site to the left and the first site to the right.
left = layers.Cropping1D(cropping=(L - 1, 0))(inputs)
right = layers.Cropping1D(cropping=(0, L - 1))(inputs)
x = layers.Concatenate(axis=1)([left, inputs, right])
x = layers.Conv1D(64, kernel_size=3, padding="valid", activation="relu")(x)
x = layers.Conv1D(64, kernel_size=1, padding="same", activation="relu")(x)
outputs = layers.Conv1D(1, kernel_size=1, padding="same", activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]  # 这里的 accuracy 是逐点 0/1 精度
)


batch_size = 64
epochs = 100
best_model_path = output_dir / "best_model.h5"

history = model.fit(
    X_train,
    y_train,
    sample_weight=site_weights,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            str(best_model_path), monitor="loss", mode="min", save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="loss", mode="min", factor=0.5, patience=4,
            min_delta=1e-5, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="loss", mode="min", patience=20, min_delta=1e-5,
        ),
        keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
    ],
)

# Load the checkpoint with the lowest recorded weighted training loss.
model = keras.models.load_model(str(best_model_path))
epochs_completed = len(history.history["loss"])
best_epoch = int(np.argmin(history.history["loss"])) + 1
# Save/evaluate the selected model, not the last epoch.

model.save_weights(str(output_dir / "dp_cnn_weights_3.weights.h5"))
model.save(str(output_dir / "model.h5"))
metrics = model.evaluate(X_test, y_test, batch_size=256, return_dict=True, verbose=0)
weighted_metrics = model.evaluate(
    X_test, y_test, sample_weight=site_weights, batch_size=256,
    return_dict=True, verbose=0,
)
report = {"seed": 42, "epochs": epochs_completed, "max_epochs": epochs,
          "best_epoch": best_epoch, "selection_metric": "weighted_training_loss",
          "best_recorded_training_loss": float(min(history.history["loss"])),
          "weighted_test_loss": float(weighted_metrics["loss"]),
          "structure_weights": {format(i, "03b"): float(w) for i, w in enumerate(structure_weights)}, "tensorflow_version": tf.__version__,
          "train_samples": len(X_train), "validation_samples": 0,
          "test_samples": len(X_test),
          "evaluation_scope": "all_samples_same_as_training",
          "test_metrics": {k: float(v) for k, v in metrics.items()}}
(output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
np.savez_compressed(output_dir / "split_indices.npz", train=train_indices,
                    validation=np.array([], dtype=np.int64), test=test_indices)
print(f"All-sample fitting metrics (same data as training): {metrics}")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, metric in zip(axes, ["loss", "accuracy"]):
    ax.plot(range(1, epochs_completed + 1), history.history[metric], label="Train")
    ax.set(xlabel="Epoch", ylabel="Weighted BCE" if metric == "loss" else "Unweighted site accuracy")
    ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "training_curves.png", dpi=180)
plt.close(fig)

# Evaluate all training samples; tolerance is an absolute probability error.
compare_local_structures(
    model, X_test, y_test,
    Path(__file__).resolve().parent / "cnn_2_results",
    tolerance=0.02,
)
