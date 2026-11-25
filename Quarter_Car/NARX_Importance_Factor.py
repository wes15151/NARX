
from pathlib import Path

import numpy as np


N_TRAIN = 50
TRAINING_SEED = 42
MAX_LAG = 4
IMPORTANCE_THRESHOLD = 1e-3
KEEP_CONSTANT = True
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


NARX_terms = make_narx_terms()


def build_narx_matrix(y, u, terms):
    indices = np.arange(MAX_LAG, len(y))
    design = np.ones((len(indices), len(terms)))
    for q, (l, m, j, k) in enumerate(terms):
        design[:, q] = y[indices - j] ** l * u[indices - k] ** m
    return design


def fit_ols(design, target):
    scales = np.linalg.norm(design, axis=0)
    scales[scales == 0] = 1.0
    beta_scaled = np.linalg.lstsq(design / scales, target, rcond=None)[0]
    return beta_scaled / scales


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

    # One coefficient vector per RK4 trajectory, using every candidate term.
    narx_coefficients = np.zeros((len(NARX_terms), N_TRAIN))
    for i in range(N_TRAIN):
        design = build_narx_matrix(y_train[:, i], u_train[:, i], NARX_terms)
        target = y_train[MAX_LAG:, i]
        narx_coefficients[:, i] = fit_ols(design, target)

    print(f"OLS coefficient matrix shape (terms, training samples): {narx_coefficients.shape}")


    # Eq. (15): normalize squared coefficients per sample, then average over samples.
    # The empty-set (constant) term is outside the equation's denominator.
    constant_mask = np.all(NARX_terms[:, :2] == 0, axis=1)
    nonconstant_mask = ~constant_mask
    nonconstant_coefficients = narx_coefficients[nonconstant_mask, :]

    if not np.all(np.isfinite(nonconstant_coefficients)):
        raise ValueError("Cannot calculate importance from nonfinite NARX coefficients.")

    # A common scale per sample cancels in the ratio and avoids square overflow.
    coefficient_scales = np.max(np.abs(nonconstant_coefficients), axis=0)
    if np.any(coefficient_scales == 0):
        raise ValueError("Importance is undefined for a sample with all nonconstant coefficients zero.")
    squared_coefficients = (nonconstant_coefficients / coefficient_scales) ** 2
    sample_importance = squared_coefficients / squared_coefficients.sum(axis=0, keepdims=True)

    importance_factors = np.full(len(NARX_terms), np.nan)
    importance_factors[nonconstant_mask] = sample_importance.mean(axis=1)

    if not np.isfinite(IMPORTANCE_THRESHOLD) or not 0 <= IMPORTANCE_THRESHOLD <= 1:
        raise ValueError("IMPORTANCE_THRESHOLD must lie between 0 and 1.")

    selected_mask = nonconstant_mask & (importance_factors >= IMPORTANCE_THRESHOLD)
    if KEEP_CONSTANT:
        selected_mask |= constant_mask

    selected_term_indices = np.flatnonzero(selected_mask)
    selected_NARX_terms = NARX_terms[selected_term_indices]
    selected_importance_factors = importance_factors[selected_term_indices]
    # These are the existing full-model OLS coefficients, not a refit of the reduced model.
    selected_narx_coefficients = narx_coefficients[selected_term_indices, :]

    print(f"\nImportance threshold: {IMPORTANCE_THRESHOLD:g}")
    print(f"Selected terms: {len(selected_NARX_terms)}/{len(NARX_terms)}")
    for term_id, index in enumerate(selected_term_indices, start=1):
        l, m, j, k = NARX_terms[index]
        factors = []
        if l:
            factor = f"y2(k-{j})"
            factors.append(factor + (f"^{l}" if l > 1 else ""))
        if m:
            factor = f"u(k-{k})" if k else "u(k)"
            factors.append(factor + (f"^{m}" if m > 1 else ""))
        expression = " * ".join(factors) if factors else "1"
        importance_text = ("constant retained separately" if constant_mask[index]
                           else f"importance={importance_factors[index]:.6e}")
        print(f"Term {term_id}: {expression}; {importance_text}")

    return {
        "time": time,
        "initial_state": initial_state,
        "training_indices": training_indices,
        "samples_train": samples_train,
        "y_train": y_train,
        "u_train": u_train,
        "all_terms": NARX_terms,
        "best_terms": selected_NARX_terms,
        "selected_term_indices": selected_term_indices,
        "importance_factors": importance_factors,
        "selected_importance_factors": selected_importance_factors,
        "full_narx_coefficients": narx_coefficients,
        "narx_coefficients": selected_narx_coefficients,
        "coefficients_refitted": False,
    }


if __name__ == "__main__":
    result = main()
