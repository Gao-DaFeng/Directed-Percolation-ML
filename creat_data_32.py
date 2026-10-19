import numpy as np
import random
from pathlib import Path
from numba import njit


ptrain = []
for i in range(51):
    ptrain.append(round(0.0 + i * 0.02, 2))


@njit
def doPercolationStep(vector, PROP):
    vector_copy = np.zeros_like(vector)
    LENGTH = len(vector)
    for i in range(LENGTH):
        if vector[i] == 1:
            pro1 = random.random()
            pro2 = random.random()
            if pro1 < PROP:
                vector_copy[(i + LENGTH - 1) % LENGTH] = 1
            if pro2 < PROP:
                vector_copy[(i + 1) % LENGTH] = 1
            # vector_copy[i] = 0
    return vector_copy
LENGTH = 32
N_RANDOM = 3000
N_ONES = 100
N_ZEROS = 100


def generate_dataset():
    """Generate aligned initial configurations, probabilities and next states."""
    rng = np.random.default_rng()
    samples_per_probability = N_RANDOM + N_ONES + N_ZEROS
    total_samples = len(ptrain) * samples_per_probability
    initial_configs = np.empty((total_samples, LENGTH), dtype=np.uint8)
    probabilities = np.repeat(np.asarray(ptrain), samples_per_probability)
    next_configs = np.empty_like(initial_configs)

    for index, p in enumerate(ptrain):
        start = index * samples_per_probability
        stop = start + samples_per_probability
        configs = initial_configs[start:stop]
        # Each random site is independently 0 or 1 with equal probability.
        configs[:N_RANDOM] = rng.integers(
            0, 2, size=(N_RANDOM, LENGTH), dtype=np.uint8
        )
        configs[N_RANDOM:N_RANDOM + N_ONES] = 1
        configs[N_RANDOM + N_ONES:] = 0
        for offset, vector in enumerate(configs):
            next_configs[start + offset] = doPercolationStep(vector, p)

    return initial_configs, probabilities, next_configs


def main():
    initial_configs, probabilities, next_configs = generate_dataset()
    output_path = Path(__file__).resolve().with_name("data_32_2.npz")
    np.savez_compressed(
        output_path,
        initial_configs=initial_configs,
        probabilities=probabilities,
        next_configs=next_configs,
    )
    print(f"Saved {len(probabilities)} samples to {output_path}")
    print(f"initial_configs: {initial_configs.shape}")
    print(f"probabilities: {probabilities.shape}")
    print(f"next_configs: {next_configs.shape}")


if __name__ == "__main__":
    main()
