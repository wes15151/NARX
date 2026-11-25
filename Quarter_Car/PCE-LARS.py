
from itertools import combinations
from pathlib import Path

import numpy as np
from UQpy.distributions import Normal, Uniform, JointIndependent
from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.PolynomialBasis import PolynomialBasis
from UQpy.surrogates.polynomial_chaos.regressions import LeastAngleRegression, LeastSquareRegression
from UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion import PolynomialChaosExpansion
from threadpoolctl import threadpool_limits

from NARX_LARS import main as build_lars_narx
from NARX_LARS import build_narx_matrix, fit_ols, MAX_LAG, DATA_PATH
from NARX_Importance_Factor import main as build_importance_narx
from quarter_car_prediction import run_population_prediction, plot_population_comparison


NARX_METHOD = "lars"  # "lars" or "importance"; only one method runs.
P_MAX = 20
INTERACTION_RANK = 2
Q_TRUNCATION = 1.0
N_PREDICT = 100000
PREDICTION_SEED = 2026
PREDICTION_BATCH_SIZE = 2000


def load_narx_result(method):
    if method == "lars":
        return build_lars_narx()
    if method == "importance":
        return build_importance_narx()
    raise ValueError('NARX_METHOD must be "lars" or "importance".')


def prepare_narx_coefficients(narx_result):
    if narx_result.get("coefficients_refitted", False):
        return narx_result["narx_coefficients"]

    # Importance selection originally slices the full-model coefficients.
    # Refit the reduced structure before using its coefficients as PCE targets.
    terms = narx_result["best_terms"]
    if len(terms) == 0:
        raise ValueError("No NARX terms were selected; revise the importance threshold.")
    y_train, u_train = narx_result["y_train"], narx_result["u_train"]
    coefficients = np.zeros((len(terms), y_train.shape[1]))
    for i in range(y_train.shape[1]):
        design = build_narx_matrix(y_train[:, i], u_train[:, i], terms)
        coefficients[:, i] = fit_ols(design, y_train[MAX_LAG:, i])
    return coefficients


def build_pce_basis(n_variables):
    if n_variables != 7:
        raise ValueError("The quarter-car input distribution requires 7 variables.")
    if INTERACTION_RANK != 2 or Q_TRUNCATION != 1:
        raise ValueError("This basis enumeration implements r=2 and q=1.")

    normal_mean = [2000.0, 2000.0, 20.0, 40.0, 600.0]
    normal_std = [200.0, 200.0, 2.0, 4.0, 60.0]
    marginals = [Normal(loc=mean, scale=std)
                 for mean, std in zip(normal_mean, normal_std)]
    marginals += [Uniform(loc=0.09, scale=0.02),
                  Uniform(loc=1.8 * np.pi, scale=0.4 * np.pi)]
    distribution = JointIndependent(marginals=marginals)

    multi_index = [np.zeros(n_variables, dtype=int)]
    for degree in range(1, P_MAX + 1):
        for variable in range(n_variables):
            alpha = np.zeros(n_variables, dtype=int)
            alpha[variable] = degree
            multi_index.append(alpha)
        for first, second in combinations(range(n_variables), 2):
            for first_degree in range(1, degree):
                alpha = np.zeros(n_variables, dtype=int)
                alpha[first] = first_degree
                alpha[second] = degree - first_degree
                multi_index.append(alpha)
    multi_index = np.array(multi_index, dtype=int)

    polynomials = PolynomialBasis.construct_arbitrary_basis(
        n_variables, distribution, multi_index
    )
    basis = PolynomialBasis(
        inputs_number=n_variables,
        polynomials_number=len(multi_index),
        multi_index_set=multi_index,
        polynomials=polynomials,
        distributions=distribution,
    )
    return basis, multi_index, polynomials, distribution


@threadpool_limits.wrap(limits=1, user_api="blas")
def fit_degree_candidates(basis, target, inputs, global_columns, degree):
    n_train = len(target)
    centered = basis - basis.mean(axis=0)
    scales = np.linalg.norm(centered, axis=0)
    usable = np.flatnonzero(scales > 1e-12)
    supports = [np.array([0], dtype=int)]
    seen = {(0,)}
    if len(usable) and np.var(target) > 0:
        lar = LeastAngleRegression(
            fit_intercept=False,
            n_nonzero_coefs=min(n_train - 2, len(usable)),
        )
        lar.run(inputs, target - target.mean(), centered[:, usable] / scales[usable])
        for path_beta in lar.Beta_path.T:
            tolerance = 1e-10 * np.max(np.abs(path_beta))
            active = usable[np.abs(path_beta) > tolerance]
            key = tuple(sorted([0, *active.tolist()]))
            if key not in seen and len(key) < n_train:
                seen.add(key)
                supports.append(np.array(key, dtype=int))

    candidates = []
    variance_sum = np.sum((target - target.mean()) ** 2)
    for support in supports:
        design = basis[:, support]
        column_scales = np.linalg.norm(design, axis=0)
        column_scales[column_scales == 0] = 1.0
        scaled = design / column_scales
        left, singular, right = np.linalg.svd(scaled, full_matrices=False)
        cutoff = np.finfo(float).eps * max(scaled.shape) * singular[0]
        if np.count_nonzero(singular > cutoff) != len(support):
            continue
        coefficients = (right.T @ ((left.T @ target) / singular)) / column_scales
        leverage = np.sum(left ** 2, axis=1)
        if np.any(1 - leverage <= 1e-10):
            continue
        residual = target - design @ coefficients
        ols_mse = float(np.mean(residual ** 2))
        loo = (float(np.sum((residual / (1 - leverage)) ** 2) / variance_sum)
               if variance_sum > 0 else 0.0)
        if np.isfinite(loo) and np.isfinite(ols_mse):
            candidates.append({
                "term_indices": global_columns[support].copy(),
                "coefficients": coefficients,
                "loo_error": loo,
                "ols_mse": ols_mse,
                "search_degree": degree,
            })

    if not candidates:
        raise ValueError(f"No valid sparse PCE candidate at degree {degree}.")
    # Stage 1: retain the minimum-LOO candidate for this degree.
    representative = min(candidates, key=lambda r: (r["loo_error"], len(r["term_indices"])))
    return candidates, representative


def train_sparse_pces(inputs, narx_coefficients):
    if len(inputs) < 3:
        raise ValueError("Sparse PCE selection requires at least 3 training samples.")
    basis, multi_index, polynomials, distribution = build_pce_basis(inputs.shape[1])
    design_all = basis.evaluate_basis(inputs)
    total_degrees = multi_index.sum(axis=1)
    models, results, candidate_histories = [], [], []
    pce_coefficients = np.zeros((len(multi_index), narx_coefficients.shape[0]))

    for coefficient_id in range(narx_coefficients.shape[0]):
        target = narx_coefficients[coefficient_id, :]
        representatives = []
        history = []
        for degree in range(1, P_MAX + 1):
            columns = np.flatnonzero(total_degrees <= degree)
            candidates, representative = fit_degree_candidates(
                design_all[:, columns], target, inputs, columns, degree
            )
            history.append({"degree": degree, "candidates": candidates,
                            "loo_best_candidate": representative})
            representatives.append(representative)

        # Stage 2: requested final criterion is OLS training MSE.
        # PCE_NARX.py instead selected the global minimum LOO across degrees.
        best = min(representatives, key=lambda r: (
            r["ols_mse"], r["loo_error"], len(r["term_indices"]), r["search_degree"]
        ))
        indices = best["term_indices"]
        sparse_basis = PolynomialBasis(
            inputs_number=inputs.shape[1],
            polynomials_number=len(indices),
            multi_index_set=multi_index[indices],
            polynomials=[polynomials[index] for index in indices],
            distributions=distribution,
        )
        model = PolynomialChaosExpansion(sparse_basis, LeastSquareRegression())
        model.set_data(inputs, target)
        model.coefficients = best["coefficients"].reshape(-1, 1)
        model.outputs_number = 1
        models.append(model)
        results.append(best)
        candidate_histories.append(history)
        pce_coefficients[indices, coefficient_id] = best["coefficients"]
        print(f"NARX coefficient {coefficient_id + 1}: "
              f"LOO={best['loo_error']:.6e}, OLS MSE={best['ols_mse']:.6e}, "
              f"PCE terms={len(indices)}, maximum degree={total_degrees[indices].max()}",
              flush=True)

    return {"models": models, "results": results, "candidate_histories": candidate_histories,
            "pce_multi_index": multi_index, "pce_coefficients": pce_coefficients,
            "selection_rule": "minimum LOO per degree, then minimum OLS training MSE"}


def main():
    narx_result = load_narx_result(NARX_METHOD)
    NARX_terms = narx_result["best_terms"]
    samples_train = narx_result["samples_train"]
    print(f"Active NARX method: {NARX_METHOD}; loaded {len(NARX_terms)} terms")
    narx_coefficients = prepare_narx_coefficients(narx_result)
    pce_result = train_sparse_pces(samples_train, narx_coefficients)

    def predict_coefficients(inputs):
        return np.column_stack([
            model.predict(inputs).reshape(-1) for model in pce_result["models"]
        ])

    population_result = run_population_prediction(
        time=narx_result["time"], initial_state=narx_result["initial_state"],
        terms=NARX_terms, predict_coefficients=predict_coefficients,
        n_samples=N_PREDICT, seed=PREDICTION_SEED, batch_size=PREDICTION_BATCH_SIZE,
        max_lag=MAX_LAG,
        output_dir=Path(__file__).resolve().parent / "results" / f"pce_lars_{NARX_METHOD}",
        model_name=f"PCE-NARX ({NARX_METHOD})", training_indices=narx_result["training_indices"],
        reference_path=DATA_PATH,
    )
    comparison_result = plot_population_comparison(
        population_result, DATA_PATH, narx_result["training_indices"],
        model_name=f"PCE-NARX ({NARX_METHOD})", batch_size=PREDICTION_BATCH_SIZE,
    )
    return {"narx_method": NARX_METHOD, "narx_result": narx_result,
            "NARX_terms": NARX_terms, "samples_train": samples_train,
            "training_indices": narx_result["training_indices"],
            "narx_coefficients": narx_coefficients, "pce_result": pce_result,
            "population_result": population_result, "comparison_result": comparison_result}


if __name__ == "__main__":
    result = main()
