
from pathlib import Path

import numpy as np
from sklearn.linear_model import lars_path


N_TRAIN = 50
TRAINING_SEED = 42
LARGE_RESPONSE_THRESHOLD = 1.2
NARX_ERROR_THRESHOLD = 1e-3
MAX_LAG = 4
DATA_PATH = (
    Path(__file__).resolve().parent.parent / "PCE_NARX" / "quarter_car_data.npz"
)


def make_narx_terms():
    # Each row: [output power, input power, output lag, input lag].
    terms = []
    for l in range(4):
        for m in range(2):
            if l + m > 3:
                continue
            y_lags = range(1, 5) if l > 0 else [0]
            u_lags = range(5) if m > 0 else [0]
            for j in y_lags:
                for k in u_lags:
                    terms.append((l, m, j, k))
    return np.array(terms, dtype=int)


def build_narx_matrix(y, u, terms):
    indices = np.arange(MAX_LAG, len(y))
    design = np.ones((len(indices), len(terms)))
    for q, (l, m, j, k) in enumerate(terms):
        design[:, q] = y[indices - j] ** l * u[indices - k] ** m
    return design


def fit_ols(design, target):
    scales = np.linalg.norm(design, axis=0)
    scales[scales == 0] = 1.0
    return np.linalg.lstsq(design / scales, target, rcond=None)[0] / scales


def recursive_narx(y_initial, u, terms, beta):
    prediction = np.zeros(len(u))
    prediction[:MAX_LAG] = y_initial[:MAX_LAG]
    l, m, j, k = terms.T
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(MAX_LAG, len(u)):
            values = prediction[n - j] ** l * u[n - k] ** m
            prediction[n] = values @ beta
            if not np.isfinite(prediction[n]):
                return None
    return prediction


def generate_structure_candidates(y_train, u_train, training_indices, all_terms):
    selected_positions = np.flatnonzero(
        np.max(np.abs(y_train), axis=0) >= LARGE_RESPONSE_THRESHOLD
    )
    constant_index = int(np.flatnonzero(np.all(all_terms[:, :2] == 0, axis=1))[0])
    candidates = []
    candidate_keys = set()
    sample_results = []

    for i in selected_positions:
        y, u = y_train[:, i], u_train[:, i]
        design_all = build_narx_matrix(y, u, all_terms)
        target = y[MAX_LAG:]
        centered = design_all - design_all.mean(axis=0)
        scales = np.linalg.norm(centered, axis=0)
        usable = np.flatnonzero(scales > 1e-12)
        if len(usable) == 0:
            continue
        _, _, coefficient_path = lars_path(
            centered[:, usable] / scales[usable], target - target.mean(), method="lar"
        )

        best_error, best_key, best_beta = np.inf, None, None
        evaluated_keys = set()
        for step in range(1, coefficient_path.shape[1]):
            coefficients = coefficient_path[:, step]
            tolerance = 1e-10 * np.max(np.abs(coefficients))
            active = usable[np.abs(coefficients) > tolerance]
            if len(active) == 0:
                continue
            key = tuple(sorted([constant_index, *active.tolist()]))
            if key in evaluated_keys:
                continue
            evaluated_keys.add(key)
            columns = np.array(key, dtype=int)
            beta = fit_ols(design_all[:, columns], target)
            prediction = recursive_narx(y, u, all_terms[columns], beta)
            if prediction is None:
                continue
            # Original per-trajectory LARS-path score: uncentered output energy.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                error = np.sum((prediction[MAX_LAG:] - target) ** 2) / np.sum(target ** 2)
            if np.isfinite(error) and error < best_error:
                best_error, best_key, best_beta = float(error), key, beta

        if best_key is None:
            continue
        sample_results.append({
            "sample_index": int(training_indices[i]),
            "term_indices": np.array(best_key, dtype=int),
            "beta": best_beta,
            "relative_squared_error": best_error,
        })
        if best_key not in candidate_keys:
            candidate_keys.add(best_key)
            candidates.append(np.array(best_key, dtype=int))

    return candidates, sample_results, selected_positions


def select_narx_model(y_train, u_train, all_terms, candidates):
    n_train = y_train.shape[1]
    evaluations = []

    for candidate_id, term_indices in enumerate(candidates, start=1):
        terms = all_terms[term_indices]
        coefficients_all = np.zeros((len(terms), n_train))
        relative_errors = np.full(n_train, np.inf)
        mse_all = np.full(n_train, np.inf)

        for i in range(n_train):
            y, u = y_train[:, i], u_train[:, i]
            # Preserve the original design-column order and common four-step lag.
            design = build_narx_matrix(y, u, all_terms)[:, term_indices]
            beta = fit_ols(design, y[MAX_LAG:])
            coefficients_all[:, i] = beta
            prediction = recursive_narx(y, u, terms, beta)
            if prediction is None:
                continue
            # Final structure score: centered, normalized squared trajectory error.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                squared_error = np.sum((y - prediction) ** 2)
                denominator = np.sum((y - y.mean()) ** 2)
                if denominator > 0 and np.isfinite(squared_error):
                    relative_errors[i] = squared_error / denominator
                    mse_all[i] = squared_error / len(y)

        evaluations.append({
            "candidate_id": candidate_id,
            "term_indices": term_indices.copy(),
            "coefficients": coefficients_all,
            "relative_errors": relative_errors,
            "mean_relative_error": float(relative_errors.mean()),
            "mean_mse": float(mse_all.mean()),
        })

    eligible = [r for r in evaluations if r["mean_relative_error"] < NARX_ERROR_THRESHOLD]
    if not eligible:
        raise RuntimeError(
            f"No NARX candidate satisfies mean recursive error < {NARX_ERROR_THRESHOLD:g}."
        )
    best = min(eligible, key=lambda r: (len(r["term_indices"]), r["mean_relative_error"]))
    return best, evaluations


def print_selected_terms(terms):
    for q, (l, m, j, k) in enumerate(terms, start=1):
        factors = []
        if l:
            factor = f"y2(k-{j})"
            factors.append(factor + (f"^{l}" if l > 1 else ""))
        if m:
            factor = f"u(k-{k})" if k else "u(k)"
            factors.append(factor + (f"^{m}" if m > 1 else ""))
        print(f"Term {q}: " + (" * ".join(factors) if factors else "1"))


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Generate the RK4 reference data first: {DATA_PATH}")
    with np.load(DATA_PATH, allow_pickle=False) as data:
        time = data["time"]
        samples = data["samples"]
        y2_all = data["y2_all"]
        initial_state = data["initial_state"]
    if len(samples) <= N_TRAIN:
        raise ValueError("The stored population must exceed the training size.")
    training_indices = np.random.default_rng(TRAINING_SEED).choice(
        len(samples), size=N_TRAIN, replace=False
    )
    samples_train = samples[training_indices]
    y_train = y2_all[:, training_indices]
    u_train = samples_train[:, 5] * np.sin(time[:, None] * samples_train[:, 6])
    all_terms = make_narx_terms()

    candidates, sample_results, selected_positions = generate_structure_candidates(
        y_train, u_train, training_indices, all_terms
    )
    best, evaluations = select_narx_model(y_train, u_train, all_terms, candidates)
    best_terms = all_terms[best["term_indices"]]
    narx_coefficients = best["coefficients"]

    print(f"Number of terms: {len(best_terms)}")
    print(f"Mean recursive error: {best['mean_relative_error']:.6e}")
    print_selected_terms(best_terms)
    print(f"OLS coefficient matrix shape (terms, training samples): {narx_coefficients.shape}")

    return {
        "time": time,
        "initial_state": initial_state,
        "training_indices": training_indices,
        "samples_train": samples_train,
        "y_train": y_train,
        "u_train": u_train,
        "all_terms": all_terms,
        "selected_indices": training_indices[selected_positions],
        "narx_candidates": candidates,
        "sample_candidate_results": sample_results,
        "candidate_evaluations": evaluations,
        "best_terms": best_terms,
        "narx_coefficients": narx_coefficients,
        "coefficients_refitted": True,
        "relative_errors": best["relative_errors"],
    }


if __name__ == "__main__":
    result = main()
