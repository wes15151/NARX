
from pathlib import Path

import numpy as np
from UQpy.distributions import Normal, Uniform, JointIndependent
from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.PolynomialBasis import PolynomialBasis

from NARX_LARS import main as build_lars_narx
from NARX_LARS import build_narx_matrix, fit_ols, MAX_LAG, DATA_PATH
from NARX_Importance_Factor import main as build_importance_narx
from quarter_car_prediction import run_population_prediction, plot_population_comparison


NARX_METHOD = "lars"  # "lars" or "importance"; only one method runs.
PDD_INTERACTION_DEGREE = 1
PDD_MAX_ORDER = 3
N_PREDICT = 100000
PREDICTION_SEED = 2026
PREDICTION_BATCH_SIZE = 2000


def load_narx_result(method):
    if method == "lars":
        return build_lars_narx()
    if method == "importance":
        return build_importance_narx()
    raise ValueError('NARX_METHOD must be "lars" or "importance".')


def build_pdd_basis(n_variables):
    if n_variables != 7:
        raise ValueError("The quarter-car input distribution requires 7 variables.")
    if PDD_INTERACTION_DEGREE != 1:
        raise ValueError("This enumeration implements univariate PDD (interaction degree 1).")

    normal_mean = [2000.0, 2000.0, 20.0, 40.0, 600.0]
    normal_std = [200.0, 200.0, 2.0, 4.0, 60.0]
    marginals = [Normal(loc=mean, scale=std)
                 for mean, std in zip(normal_mean, normal_std)]
    marginals += [Uniform(loc=0.09, scale=0.02),
                  Uniform(loc=1.8 * np.pi, scale=0.4 * np.pi)]
    distribution = JointIndependent(marginals=marginals)

    # Constant plus univariate orders 1, 2, 3; no cross-variable products.
    multi_index = [np.zeros(n_variables, dtype=int)]
    for variable in range(n_variables):
        for order in range(1, PDD_MAX_ORDER + 1):
            alpha = np.zeros(n_variables, dtype=int)
            alpha[variable] = order
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
    return basis, multi_index


def prepare_narx_coefficients(narx_result):
    if narx_result.get("coefficients_refitted", False):
        return narx_result["narx_coefficients"]

    # Match PCE-LARS.py: refit the selected importance-based NARX structure.
    terms = narx_result["best_terms"]
    if len(terms) == 0:
        raise ValueError("No NARX terms were selected; revise the importance threshold.")
    y_train, u_train = narx_result["y_train"], narx_result["u_train"]
    coefficients = np.zeros((len(terms), y_train.shape[1]))
    for i in range(y_train.shape[1]):
        design = build_narx_matrix(y_train[:, i], u_train[:, i], terms)
        coefficients[:, i] = fit_ols(design, y_train[MAX_LAG:, i])
    return coefficients


def fit_pdd_coefficients(design, narx_coefficients):
    """Section 4.4, Eqs. (35)-(38): standard least-squares PDD regression."""
    # A: (training samples, PDD terms); B: (training samples, NARX terms).
    targets = narx_coefficients.T
    n_samples, n_terms = design.shape
    if targets.ndim != 2 or targets.shape[0] != n_samples:
        raise ValueError("PDD basis and NARX coefficients must use the same training samples.")
    if n_samples <= n_terms:
        raise ValueError(
            f"Section 4.4 requires more training samples than PDD terms: "
            f"received {n_samples} samples and {n_terms} terms."
        )
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(targets)):
        raise ValueError("PDD regression requires finite basis values and NARX coefficients.")

    scales = np.linalg.norm(design, axis=0)
    if np.any(scales == 0):
        raise ValueError("A PDD basis column is zero at every training input.")
    # Solve A C = B without forming the normal-equation inverse.
    scaled_coefficients, _, rank, singular_values = np.linalg.lstsq(
        design / scales, targets, rcond=None
    )
    if rank < n_terms:
        raise ValueError("The PDD design is rank deficient; the SLS coefficients are not unique.")

    coefficients = scaled_coefficients / scales[:, None]
    fitted_targets = design @ coefficients
    mse = np.mean((targets - fitted_targets) ** 2, axis=0)
    scaled_condition_number = float(singular_values[0] / singular_values[-1])
    return coefficients, fitted_targets, mse, scaled_condition_number


if __name__ == "__main__":
    narx_result = load_narx_result(NARX_METHOD)
    NARX_terms = narx_result["best_terms"]
    samples_train = narx_result["samples_train"]
    training_indices = narx_result["training_indices"]
    print(f"Active NARX method: {NARX_METHOD}; loaded {len(NARX_terms)} terms")

    PDD_basis, PDD_multi_index = build_pdd_basis(samples_train.shape[1])
    PDD_design_train = PDD_basis.evaluate_basis(samples_train)
    print(f"PDD interaction degree: {PDD_INTERACTION_DEGREE}")
    print(f"PDD maximum order: {PDD_MAX_ORDER}")
    print(f"Number of PDD basis terms: {len(PDD_multi_index)}")
    print(f"PDD design matrix shape (samples, terms): {PDD_design_train.shape}")

    narx_coefficients = prepare_narx_coefficients(narx_result)
    (PDD_coefficients, NARX_coefficients_PDD_train,
     PDD_training_mse, PDD_scaled_condition_number) = fit_pdd_coefficients(
        PDD_design_train, narx_coefficients
    )
    print(f"PDD coefficient matrix shape (PDD terms, NARX terms): {PDD_coefficients.shape}")
    print(f"Column-scaled PDD design condition number: {PDD_scaled_condition_number:.6e}")
    for coefficient_id, mse in enumerate(PDD_training_mse, start=1):
        print(f"NARX coefficient {coefficient_id}: PDD training MSE={mse:.6e}")

    def predict_coefficients(inputs):
        return PDD_basis.evaluate_basis(inputs) @ PDD_coefficients

    population_result = run_population_prediction(
        time=narx_result["time"], initial_state=narx_result["initial_state"],
        terms=NARX_terms, predict_coefficients=predict_coefficients,
        n_samples=N_PREDICT, seed=PREDICTION_SEED, batch_size=PREDICTION_BATCH_SIZE,
        max_lag=MAX_LAG,
        output_dir=Path(__file__).resolve().parent / "results" / f"pdd_{NARX_METHOD}",
        model_name=f"PDD-NARX ({NARX_METHOD})", training_indices=training_indices,
        reference_path=DATA_PATH,
    )
    comparison_result = plot_population_comparison(
        population_result, DATA_PATH, training_indices,
        model_name=f"PDD-NARX ({NARX_METHOD})", batch_size=PREDICTION_BATCH_SIZE,
    )
