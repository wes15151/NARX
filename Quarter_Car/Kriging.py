
from pathlib import Path

import numpy as np
import warnings
from scipy.linalg import cho_solve, cholesky
from scipy.optimize import minimize, OptimizeResult

import NARX_LARS as lars_narx
import NARX_Importance_Factor as importance_narx

from NARX_LARS import main as build_lars_narx
from NARX_Importance_Factor import main as build_importance_narx
from quarter_car_prediction import run_population_prediction, plot_population_comparison


NARX_METHOD = "importance"  # "lars" or "importance"; only one method runs.
RBF_LENGTH_SCALE = 1.0  # Initial value in standardized input coordinates.
RBF_SIGNAL_VARIANCE = 1.0  # sigma_f**2; not yet optimized.
JITTER_RELATIVE = 1e-10
N_PREDICT = 100000
PREDICTION_SEED = 2026
PREDICTION_BATCH_SIZE = 2000


def load_narx_result(method):
    if method == "lars":
        return build_lars_narx()
    if method == "importance":
        return build_importance_narx()
    raise ValueError('NARX_METHOD must be "lars" or "importance".')


def fit_coefficient_matrix(narx_result, method):
    """Refit the selected structure: a[j, i] is term j's coefficient for sample i."""
    if method == "lars":
        narx_module = lars_narx
    elif method == "importance":
        narx_module = importance_narx
    else:
        raise ValueError('NARX_METHOD must be "lars" or "importance".')

    terms = narx_result["best_terms"]
    y_train = narx_result["y_train"]
    u_train = narx_result["u_train"]
    a = np.empty((len(terms), y_train.shape[1]))

    for i in range(y_train.shape[1]):
        # Use RK4 output history to construct the OLS design matrix.
        Phi = narx_module.build_narx_matrix(y_train[:, i], u_train[:, i], terms)
        target = y_train[narx_module.MAX_LAG:, i]
        # Scaled least squares; coefficients are returned in the original units.
        a[:, i] = narx_module.fit_ols(Phi, target)

    return a


def rbf_kernel(inputs, length_scale=1.0, signal_variance=1.0, other_inputs=None):
    """Return training covariance or cross-covariance with other inputs."""
    if other_inputs is None:
        other_inputs = inputs
    differences = (inputs[:, None, :] - other_inputs[None, :, :]) / length_scale
    squared_distances = np.sum(differences ** 2, axis=2)
    return signal_variance * np.exp(-0.5 * squared_distances)


def negative_log_marginal_likelihood(log_parameters, inputs, centered_target):
    """Evaluate -log p(target | inputs) using a Cholesky decomposition."""
    length_scale, signal_variance = np.exp(log_parameters)
    covariance = rbf_kernel(inputs, length_scale, signal_variance)
    covariance += JITTER_RELATIVE * signal_variance * np.eye(len(inputs))
    try:
        lower = cholesky(covariance, lower=True)
        alpha = cho_solve((lower, True), centered_target)
    except np.linalg.LinAlgError:
        return np.inf

    # log|C| = 2 * sum(log(diag(L))), where C = L @ L.T.
    value = (
        0.5 * centered_target @ alpha
        + np.log(np.diag(lower)).sum()
        + 0.5 * len(inputs) * np.log(2.0 * np.pi)
    )
    return float(value) if np.isfinite(value) else np.inf


def optimize_gp_hyperparameters(inputs, coefficients, means):
    """Fit one independent GP per row of the NARX coefficient matrix."""
    models = []
    for j in range(coefficients.shape[0]):
        centered = coefficients[j, :] - means[j]
        # Scale the target during optimization to handle different coefficient units.
        output_scale = float(np.std(centered))
        if output_scale == 0.0:
            output_scale = 1.0
        normalized_target = centered / output_scale

        # Optimize positive parameters in log coordinates. Signal variance here
        # is in normalized output units; convert it back below.
        # For fixed length, the ML signal variance is r.T @ R^-1 @ r / N.
        # Screen lengths before local refinement so a stalled optimizer cannot
        # discard a better likelihood already found on the grid.
        starts = []
        for length in np.logspace(-3, 3, 49):
            correlation = rbf_kernel(inputs, length)
            correlation.flat[::len(inputs) + 1] += JITTER_RELATIVE
            lower = cholesky(correlation, lower=True)
            quadratic = normalized_target @ cho_solve((lower, True), normalized_target)
            variance = np.clip(quadratic / len(inputs), 1e-8, 1e4)
            nll = (quadratic / variance + len(inputs) * np.log(2 * np.pi * variance)) / 2
            starts.append((nll + np.log(lower.diagonal()).sum(), length, variance))
        grid_nll, length, variance = min(starts)
        grid_start = np.log([length, variance])
        optimization = minimize(
            negative_log_marginal_likelihood,
            x0=grid_start,
            args=(inputs, normalized_target),
            method="L-BFGS-B",
            bounds=[(np.log(1e-3), np.log(1e3)),
                    (np.log(1e-8), np.log(1e4))],
            options={"maxiter": 500},
        )
        if not np.isfinite(optimization.fun) or optimization.fun > grid_nll:
            optimization = OptimizeResult(x=grid_start, fun=grid_nll, success=True,
                                          message="Retained best grid likelihood")
        if not optimization.success:
            warnings.warn(
                f"Coefficient {j + 1}: optimizer did not converge: "
                f"{optimization.message}", RuntimeWarning,
            )

        length_scale, normalized_variance = np.exp(optimization.x)
        signal_variance = normalized_variance * output_scale ** 2
        K_optimized = rbf_kernel(inputs, length_scale, signal_variance)
        jitter_optimized = JITTER_RELATIVE * signal_variance
        K_stable_optimized = K_optimized + jitter_optimized * np.eye(len(inputs))
        lower = cholesky(K_stable_optimized, lower=True)
        alpha = cho_solve((lower, True), centered)
        # Convert the optimized likelihood back to original coefficient units.
        log_likelihood = -float(optimization.fun) - len(inputs) * np.log(output_scale)
        models.append({
            "mean": means[j],
            "length_scale": length_scale,
            "signal_variance": signal_variance,
            "jitter": jitter_optimized,
            "K": K_optimized,
            "K_stable": K_stable_optimized,
            "cholesky": lower,
            "alpha": alpha,
            "log_marginal_likelihood": log_likelihood,
            "optimization": optimization,
        })
        print(
            f"NARX coefficient {j + 1}: length scale={length_scale:.6g}, "
            f"signal variance={signal_variance:.6g}, "
            f"log marginal likelihood={log_likelihood:.6g}"
        )
    return models


def predict_narx_coefficients(new_inputs, models, training_inputs_scaled,
                              input_mean, input_std):
    """Return GP mean coefficients with shape (new samples, NARX terms)."""
    new_inputs = np.asarray(new_inputs, dtype=float)
    if new_inputs.ndim == 1:
        new_inputs = new_inputs[None, :]
    if new_inputs.ndim != 2 or new_inputs.shape[1] != len(input_mean):
        raise ValueError("new_inputs must have shape (n_samples, n_variables).")
    if not np.all(np.isfinite(new_inputs)):
        raise ValueError("new_inputs must contain only finite values.")

    # Use the same input transformation as during GP training.
    new_inputs_scaled = (new_inputs - input_mean) / input_std
    predicted_a = np.empty((len(new_inputs_scaled), len(models)))
    for j, model in enumerate(models):
        # Shape: (training samples, new samples). No jitter on cross-covariance.
        k_star = rbf_kernel(
            training_inputs_scaled,
            length_scale=model["length_scale"],
            signal_variance=model["signal_variance"],
            other_inputs=new_inputs_scaled,
        )
        # alpha = C_j^{-1} (a_j - m_j * 1), already solved during fitting.
        predicted_a[:, j] = model["mean"] + k_star.T @ model["alpha"]
    return predicted_a


if __name__ == "__main__":
    narx_result = load_narx_result(NARX_METHOD)
    NARX_terms = narx_result["best_terms"]
    samples_train = narx_result["samples_train"]
    training_indices = narx_result["training_indices"]
    print(f"Active NARX method: {NARX_METHOD}; loaded {len(NARX_terms)} terms")
    a = fit_coefficient_matrix(narx_result, NARX_METHOD)
    print(f"NARX coefficient matrix a shape (terms, training samples): {a.shape}")
    # Average each term's coefficients over the training samples.
    m = a.mean(axis=1)
    print(f"Mean coefficient vector m shape: {m.shape}")

    # Standardize physical inputs so different units do not dominate distances.
    input_mean = samples_train.mean(axis=0)
    input_std = samples_train.std(axis=0)
    input_std[input_std == 0.0] = 1.0
    samples_train_scaled = (samples_train - input_mean) / input_std
    K = rbf_kernel(
        samples_train_scaled,
        length_scale=RBF_LENGTH_SCALE,
        signal_variance=RBF_SIGNAL_VARIANCE,
    )
    print(f"RBF kernel matrix K shape: {K.shape}")
    # Numerical stabilization only; keep the original covariance matrix K.
    jitter = JITTER_RELATIVE * RBF_SIGNAL_VARIANCE
    K_stable = K + jitter * np.eye(K.shape[0])

    gp_models = optimize_gp_hyperparameters(samples_train_scaled, a, m)

    def predict_coefficients(new_inputs):
        return predict_narx_coefficients(
            new_inputs, gp_models, samples_train_scaled, input_mean, input_std
        )

    narx_module = lars_narx if NARX_METHOD == "lars" else importance_narx
    population_result = run_population_prediction(
        time=narx_result["time"], initial_state=narx_result["initial_state"],
        terms=NARX_terms, predict_coefficients=predict_coefficients,
        n_samples=N_PREDICT, seed=PREDICTION_SEED, batch_size=PREDICTION_BATCH_SIZE,
        max_lag=narx_module.MAX_LAG,
        output_dir=Path(__file__).resolve().parent / "results" / f"kriging_{NARX_METHOD}",
        model_name=f"GP-NARX ({NARX_METHOD})", training_indices=training_indices,
        reference_path=narx_module.DATA_PATH,
    )
    comparison_result = plot_population_comparison(
        population_result, narx_module.DATA_PATH, training_indices,
        model_name=f"GP-NARX ({NARX_METHOD})", batch_size=PREDICTION_BATCH_SIZE,
    )
