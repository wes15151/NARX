"""Compare LHS 100 with random and adaptive enrichment of a shared LHS 50.

Stored RK4 responses are accessed for fitting only after an input is selected.
Acquisition uses GP variance and recursive sensitivities on initial training inputs.
"""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtri
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

import Kriging as kriging
import NARX_LARS as narx
from compare_six_models import calculate_time_rmse
from quarter_car_prediction import initial_y2_history, load_rk4_step, run_population_prediction


N_INITIAL = 50
N_FINAL = 100
CANDIDATE_POOL_SIZE = 10000
N_PREDICT = 100000
SEEDS = range(1, 11)
BATCH_SIZE = 2000
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "kriging_lhs100_lhs_random_lhs_adaptive_late"
NORMAL_MEAN = np.array([2000.0, 2000.0, 20.0, 40.0, 600.0])
NORMAL_STD = np.array([200.0, 200.0, 2.0, 4.0, 60.0])
AMPLITUDE_BOUNDS = (0.09, 0.11)
OMEGA_BOUNDS = (1.8 * np.pi, 2.2 * np.pi)
PDF_EXPONENT = 0.5
SENSITIVITY_UPDATE_INTERVAL = 10
SENSITIVITY_START_TIME = 5.0


def fit_one_trajectory(index, inputs, responses, time, terms):
    y = responses[:, index]
    u = inputs[index, 5] * np.sin(inputs[index, 6] * time)
    design = narx.build_narx_matrix(y, u, terms)
    return narx.fit_ols(design, y[narx.MAX_LAG:])


@threadpool_limits.wrap(limits=1, user_api="blas")
def fit_gp(inputs, a, input_mean, input_std):
    scaled = (inputs - input_mean) / input_std
    # Keep one progress line per enrichment step; optimizer warnings remain visible.
    with redirect_stdout(StringIO()):
        models = kriging.optimize_gp_hyperparameters(scaled, a, a.mean(axis=1))
    return {"models": models, "scaled_inputs": scaled,
            "input_mean": input_mean, "input_std": input_std}


def predict_coefficients(fitted, inputs):
    return kriging.predict_narx_coefficients(
        inputs, fitted["models"], fitted["scaled_inputs"],
        fitted["input_mean"], fitted["input_std"]
    )


def recursive_sensitivity_weights(inputs, time, terms, coefficients, initial_history):
    """Average squared total derivatives over times at or after the configured start."""
    n_terms = len(terms)
    n_inputs = len(inputs)
    lag = narx.MAX_LAG
    prediction = np.zeros((len(time), n_inputs))
    prediction[:lag] = initial_history
    sensitivity = np.zeros((len(time), n_inputs, n_terms))
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(lag, len(time)):
            for q, (l, m, j, k) in enumerate(terms):
                road = inputs[:, 5] * np.sin(inputs[:, 6] * time[n - k])
                input_factor = road ** m
                phi = prediction[n - j] ** l * input_factor
                prediction[n] += coefficients[:, q] * phi
                # Direct coefficient derivative.
                sensitivity[n, :, q] += phi
                if l > 0:
                    # Chain rule through earlier predictions, including all coefficients.
                    derivative = coefficients[:, q] * l * prediction[n - j] ** (l - 1) * input_factor
                    sensitivity[n] += derivative[:, None] * sensitivity[n - j]
    if not np.isfinite(prediction).all() or not np.isfinite(sensitivity).all():
        raise RuntimeError("Representative trajectories or recursive sensitivities diverged.")
    selected_times = np.asarray(time)[lag:] >= SENSITIVITY_START_TIME
    if not np.any(selected_times):
        raise ValueError("Sensitivity start time leaves no prediction times to average.")
    late_sensitivity = sensitivity[lag:][selected_times]
    # A common scaling preserves relative mean-square weights and prevents overflow.
    scale = np.max(np.abs(late_sensitivity))
    if scale == 0.0:
        raise RuntimeError("All recursive sensitivities are zero.")
    weights = np.mean((late_sensitivity / scale) ** 2, axis=(0, 1))
    return weights / weights.max()


@threadpool_limits.wrap(limits=1, user_api="blas")
def acquisition_scores(fitted, candidate_inputs, sensitivity_weights,
                       pdf_exponent=PDF_EXPONENT, variance_exponent=1.0, weight_exponent=1.0):
    """Joint PDF**0.5 times sensitivity-weighted coefficient posterior variances."""
    scores = np.zeros(len(candidate_inputs))
    for start in range(0, len(candidate_inputs), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(candidate_inputs))
        scaled = (candidate_inputs[start:stop] - fitted["input_mean"]) / fitted["input_std"]
        for j, model in enumerate(fitted["models"]):
            weight = sensitivity_weights[j] ** weight_exponent
            if weight == 0.0:
                continue
            k_star = kriging.rbf_kernel(
                fitted["scaled_inputs"], model["length_scale"],
                model["signal_variance"], other_inputs=scaled
            )
            v = solve_triangular(model["cholesky"], k_star, lower=True)
            variance = model["signal_variance"] - np.sum(v ** 2, axis=0)
            tolerance = 1e-7 * model["signal_variance"]
            if np.any(variance < -tolerance):
                raise RuntimeError("Negative GP posterior variance exceeds rounding tolerance.")
            scores[start:stop] += weight * np.maximum(variance, 0.0) ** variance_exponent
    if not np.isfinite(scores).all():
        raise RuntimeError("Acquisition scores contain nonfinite values.")

    # Independent input model used to generate the quarter-car population.
    standardized = (candidate_inputs[:, :5] - NORMAL_MEAN) / NORMAL_STD
    log_joint_pdf = -0.5 * np.sum(standardized ** 2, axis=1)
    log_joint_pdf -= np.sum(np.log(NORMAL_STD * np.sqrt(2.0 * np.pi)))
    in_uniform_support = (
        (candidate_inputs[:, 5] >= AMPLITUDE_BOUNDS[0])
        & (candidate_inputs[:, 5] <= AMPLITUDE_BOUNDS[1])
        & (candidate_inputs[:, 6] >= OMEGA_BOUNDS[0])
        & (candidate_inputs[:, 6] <= OMEGA_BOUNDS[1])
    )
    log_joint_pdf += -np.log(AMPLITUDE_BOUNDS[1] - AMPLITUDE_BOUNDS[0])
    log_joint_pdf += -np.log(OMEGA_BOUNDS[1] - OMEGA_BOUNDS[0])
    log_joint_pdf[~in_uniform_support] = -np.inf

    # Subtracting the maximum preserves candidate ranking and avoids underflow.
    pdf_weight = np.zeros(len(log_joint_pdf))
    pdf_weight[in_uniform_support] = np.exp(
        pdf_exponent * (log_joint_pdf[in_uniform_support] - np.max(log_joint_pdf)))
    weighted_scores = scores * pdf_weight
    if not np.isfinite(weighted_scores).all() or np.max(weighted_scores) <= 0.0:
        raise RuntimeError("PDF-weighted acquisition scores are invalid.")
    return weighted_scores, scores, pdf_weight


def evaluate_seed(results, reference, all_inputs, training_union, initial_seed, output_dir):
    first = results[0]["population"]
    time, indices = first["time"], first["reference_indices"]
    held_out = ~np.isin(indices, training_union)
    common = held_out.copy()
    for start in range(0, len(indices), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(indices))
        values = reference[:, indices[start:stop]]
        common[start:stop] &= np.isfinite(values).all(axis=0)
    for result in results:
        population = result["population"]
        if (not np.array_equal(indices, population["reference_indices"])
                or not np.array_equal(population["samples"], all_inputs[indices])
                or not np.array_equal(time, population["time"])):
            raise ValueError("Models must use identical paired RK4 inputs and time grids.")
        result["finite"] = ~population["failed"].copy()
        common &= result["finite"]
    if common.sum() < 2:
        raise RuntimeError("Too few common finite held-out trajectories to compare RMSE.")

    counts = "; ".join(
        f"{r['label']}: {np.count_nonzero(held_out & ~r['finite']):,}"
        for r in results
    )
    print(f"Seed {initial_seed}: common finite held-out {common.sum():,}/{held_out.sum():,}. "
          f"Excluded nonfinite: {counts}", flush=True)
    for result in results:
        result["rmse"] = calculate_time_rmse(
            result["population"]["prediction"], reference, indices, common
        )
        print(f"{result['label']}: nonfinite trajectories "
              f"{np.count_nonzero(~result['finite']):,}/{len(indices):,}", flush=True)
        if not np.isfinite(result["rmse"]).all():
            print(f"{result['label']}: RMSE exceeds numerical range at some times.")
    labels = [r["label"] for r in results]
    rmse = np.stack([r["rmse"] for r in results])
    np.savez(output_dir / "comparison_statistics.npz", time=time, initial_seed=initial_seed,
             labels=labels, reference_indices=indices,
             common_finite_mask=common, held_out_mask=held_out,
             model_finite=np.stack([r["finite"] for r in results]),
             time_rmse=rmse)
    return labels, rmse


def lhs_inputs(count, rng):
    unit = qmc.LatinHypercube(d=7, seed=rng).random(count)
    physical = NORMAL_MEAN + NORMAL_STD * ndtri(unit[:, :5])
    forcing = qmc.scale(unit[:, 5:], [AMPLITUDE_BOUNDS[0], OMEGA_BOUNDS[0]],
                        [AMPLITUDE_BOUNDS[1], OMEGA_BOUNDS[1]])
    return np.column_stack((physical, forcing))


def train_baseline(x_train, y_train, time, indices):
    u_train = x_train[:, 5] * np.sin(time[:, None] * x_train[:, 6])
    all_terms = narx.make_narx_terms()
    print(f"Selecting NARX structure with {len(indices)} samples...", flush=True)
    candidates, _, _ = narx.generate_structure_candidates(y_train, u_train, indices, all_terms)
    best, _ = narx.select_narx_model(y_train, u_train, all_terms, candidates)
    terms = all_terms[best["term_indices"]]
    narx.print_selected_terms(terms)
    a = kriging.fit_coefficient_matrix(
        {"best_terms": terms, "y_train": y_train, "u_train": u_train}, "lars"
    )
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std[std == 0] = 1.0
    scaled = (x_train - mean) / std
    models = kriging.optimize_gp_hyperparameters(scaled, a, a.mean(axis=1))
    return terms, a, {"models": models, "scaled_inputs": scaled,
                      "input_mean": mean, "input_std": std}


def run_seed(initial_seed, inputs, responses, time, initial_state):
    if not 1 < N_INITIAL < N_FINAL:
        raise ValueError("Require 1 < N_INITIAL < N_FINAL.")
    output_dir = OUTPUT_DIR / f"seed_{initial_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if N_PREDICT > len(inputs) or CANDIDATE_POOL_SIZE > len(inputs):
        raise ValueError("Not enough stored RK4 inputs; reduce prediction/candidate counts.")
    if CANDIDATE_POOL_SIZE < N_FINAL - N_INITIAL:
        raise ValueError("Candidate pool is too small to reach N_FINAL.")
    initial_rng, lhs_rng, pool_rng, random_rng = [
        np.random.default_rng(stream) for stream in np.random.SeedSequence(initial_seed).spawn(4)
    ]
    candidate_pool = pool_rng.choice(
        len(inputs), CANDIDATE_POOL_SIZE, replace=False
    )
    random_indices = random_rng.choice(candidate_pool, N_FINAL - N_INITIAL, replace=False)
    pool = candidate_pool.copy()
    initial_inputs = lhs_inputs(N_INITIAL, initial_rng)
    lhs100 = lhs_inputs(N_FINAL, lhs_rng)
    # Evaluate the exact LHS designs on the same RK4 time grid as the stored data.
    rk4_y = initial_y2_history(np.vstack((initial_inputs, lhs100)), time, initial_state,
                               len(time), load_rk4_step())
    terms, a_initial, initial_gp = train_baseline(
        initial_inputs, rk4_y[:, :N_INITIAL], time, np.arange(N_INITIAL))
    lhs_terms, a_lhs, lhs_gp = train_baseline(
        lhs100, rk4_y[:, N_INITIAL:], time, np.arange(N_FINAL))
    input_mean, input_std = initial_gp["input_mean"], initial_gp["input_std"]
    random_inputs = np.vstack((initial_inputs, inputs[random_indices]))
    extra_coefficients = [fit_one_trajectory(i, inputs, responses, time, terms)
                          for i in random_indices]
    a_random = np.column_stack([a_initial, *extra_coefficients])
    random_gp = fit_gp(random_inputs, a_random, input_mean, input_std)

    a_adaptive = a_initial.copy()
    adaptive_gp = initial_gp
    representative_inputs = initial_inputs
    initial_history = rk4_y[:narx.MAX_LAG, :N_INITIAL]
    weights = recursive_sensitivity_weights(
        representative_inputs, time, terms,
        predict_coefficients(adaptive_gp, representative_inputs), initial_history
    )
    weight_update_sizes = [N_INITIAL]
    weight_updates = [weights.copy()]
    acquisition_weights = []
    added_indices, selected_scores = [], []
    selected_variance_scores, selected_pdf_weights = [], []
    for size in range(N_INITIAL, N_FINAL):
        # Candidate scoring uses the representative sensitivity weights cached below.
        scores, variance_scores, pdf_weights = acquisition_scores(
            adaptive_gp, inputs[pool], weights
        )
        acquisition_weights.append(weights.copy())
        selected_position = int(np.argmax(scores))
        selected_index = int(pool[selected_position])
        selected_score = float(scores[selected_position])
        selected_variance_score = float(variance_scores[selected_position])
        selected_pdf_weight = float(pdf_weights[selected_position])
        new_a = fit_one_trajectory(selected_index, inputs, responses, time, terms)
        a_adaptive = np.column_stack((a_adaptive, new_a))
        added_indices.append(selected_index)
        selected_scores.append(selected_score)
        selected_variance_scores.append(selected_variance_score)
        selected_pdf_weights.append(selected_pdf_weight)
        pool = np.delete(pool, selected_position)
        adaptive_inputs = np.vstack((initial_inputs, inputs[added_indices]))
        adaptive_gp = fit_gp(adaptive_inputs, a_adaptive, input_mean, input_std)
        if (size + 1 - N_INITIAL) % SENSITIVITY_UPDATE_INTERVAL == 0:
            weights = recursive_sensitivity_weights(
                representative_inputs, time, terms,
                predict_coefficients(adaptive_gp, representative_inputs), initial_history
            )
            weight_update_sizes.append(size + 1)
            weight_updates.append(weights.copy())
            print(f"Updated recursive sensitivity weights at N={size + 1}: {weights}", flush=True)
        print(f"Adaptive {size + 1}/{N_FINAL}: added RK4 index={selected_index}, "
              f"weighted score={selected_score:.6e}, "
              f"variance score={selected_variance_score:.6e}, "
              f"PDF weight={selected_pdf_weight:.6e}", flush=True)
        np.savez(output_dir / "sampling_history.npz", initial_lhs_inputs=initial_inputs,
                 lhs_inputs=lhs100, random_added_indices=random_indices, random_inputs=random_inputs,
                 candidate_indices=candidate_pool, adaptive_inputs=adaptive_inputs,
                 added_indices=np.array(added_indices), scores=np.array(selected_scores),
                 variance_scores=np.array(selected_variance_scores),
                 pdf_weights=np.array(selected_pdf_weights),
                 pdf_exponent=PDF_EXPONENT,
                 sensitivity_update_interval=SENSITIVITY_UPDATE_INTERVAL,
                 sensitivity_start_time=SENSITIVITY_START_TIME,
                 sensitivity_update_sizes=np.array(weight_update_sizes),
                 sensitivity_weights=np.array(weight_updates),
                 acquisition_weights=np.array(acquisition_weights),
                 candidate_pool_size=CANDIDATE_POOL_SIZE, initial_seed=initial_seed,
                 prediction_seed=initial_seed, adaptive_terms=terms, random_terms=terms,
                 lhs_terms=lhs_terms, a_lhs=a_lhs, a_random=a_random, a_adaptive=a_adaptive,
                 acquisition="normalized joint PDF**0.5 times sum of mean-square recursive sensitivity weights times coefficient GP variances",
                 structure="LHS 100 selects independently; both enrichment models retain the shared initial LHS structure")

    results = []
    for key, label, fitted, train_indices, model_terms in (
        ("lhs", f"LHS {N_FINAL}", lhs_gp, np.empty(0, dtype=int), lhs_terms),
        ("lhs_random", f"LHS {N_INITIAL} + random {N_FINAL - N_INITIAL}", random_gp,
         random_indices, terms),
        ("adaptive", f"LHS {N_INITIAL} + adaptive {N_FINAL - N_INITIAL}", adaptive_gp,
         np.array(added_indices), terms),
    ):
        population = run_population_prediction(
            time=time, initial_state=initial_state, terms=model_terms,
            predict_coefficients=lambda x, fitted=fitted: predict_coefficients(fitted, x),
            n_samples=N_PREDICT, seed=initial_seed, batch_size=BATCH_SIZE,
            max_lag=narx.MAX_LAG, output_dir=output_dir / key,
            model_name=label, training_indices=train_indices, reference_path=narx.DATA_PATH,
        )
        results.append({"label": label, "population": population})
    # Exact LHS inputs are external to the stored reference population.
    training_union = np.union1d(random_indices, added_indices)
    return evaluate_seed(results, responses, inputs, training_union, initial_seed, output_dir)


def main():
    with np.load(narx.DATA_PATH, allow_pickle=False) as data:
        inputs, responses = data["samples"], data["y2_all"]
        time, initial_state = data["time"], data["initial_state"]
    seed_errors = []
    for seed in SEEDS:
        print(f"Running comparison with seed {seed}", flush=True)
        labels, rmse = run_seed(seed, inputs, responses, time, initial_state)
        seed_errors.append(rmse)

    # Each seed contributes one RMSE curve per model with equal weight.
    mean_rmse = np.mean(seed_errors, axis=0)
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, curve, style in zip(labels, mean_rmse, (":", "--", "-.")):
        ax.plot(time, curve, linestyle=style, label=label)
    ax.set(title=f"Mean y2 RMSE against RK4 across {len(SEEDS)} seeds",
           xlabel="Time (s)", ylabel="Mean RMSE (m)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mean_time_rmse.png", dpi=180)
    fig.savefig(OUTPUT_DIR / "mean_time_rmse.pdf")
    plt.show()


if __name__ == "__main__":
    main()
