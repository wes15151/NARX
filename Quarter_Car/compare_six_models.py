"""Compare LARS-selected NARX with three coefficient surrogates."""
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import Kriging as kriging
import NARX_LARS as lars_narx
import PDD as pdd
from quarter_car_prediction import run_population_prediction


NARX_METHODS = ("lars",)
COEFFICIENT_METHODS = ("kriging", "pce_lars", "pdd")
N_PREDICT = 100000
PREDICTION_SEED = 2026
BATCH_SIZE = 2000
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "three_model_comparison"


def load_pce_module():
    # PCE-LARS.py contains a hyphen, so load it by its file path.
    path = Path(__file__).resolve().parent / "PCE-LARS.py"
    spec = importlib.util.spec_from_file_location("quarter_car_pce_lars", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_coefficient_predictor(method, inputs, a, pce_module):
    """Use the existing standalone models' fitting and prediction functions."""
    if method == "kriging":
        input_mean = inputs.mean(axis=0)
        input_std = inputs.std(axis=0)
        input_std[input_std == 0.0] = 1.0
        scaled = (inputs - input_mean) / input_std
        models = kriging.optimize_gp_hyperparameters(scaled, a, a.mean(axis=1))

        def predict(new_inputs):
            return kriging.predict_narx_coefficients(
                new_inputs, models, scaled, input_mean, input_std
            )

        return predict

    if method == "pce_lars":
        fitted = pce_module.train_sparse_pces(inputs, a)

        def predict(new_inputs):
            return np.column_stack([
                model.predict(new_inputs).reshape(-1) for model in fitted["models"]
            ])

        return predict

    if method == "pdd":
        basis, _ = pdd.build_pdd_basis(inputs.shape[1])
        design = basis.evaluate_basis(inputs)
        coefficients, _, _, _ = pdd.fit_pdd_coefficients(design, a)

        def predict(new_inputs):
            return basis.evaluate_basis(new_inputs) @ coefficients

        return predict

    raise ValueError(f"Unknown coefficient method: {method}")


def extract_qoi(trajectories, time_indices):
    """Extract t=5, t=10, max|y2| without copying all trajectories at once."""
    n_samples = trajectories.shape[1]
    qoi = np.full((3, n_samples), np.nan)
    finite = np.zeros(n_samples, dtype=bool)
    for start in range(0, n_samples, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, n_samples)
        values = trajectories[:, start:stop]
        finite[start:stop] = np.isfinite(values).all(axis=0)
        qoi[:2, start:stop] = values[time_indices, :]
        qoi[2, start:stop] = np.max(np.abs(values), axis=0)
    return qoi, finite


def evaluate_density(values, grid):
    # Rescaling keeps KDE covariance calculations safe for large finite outputs.
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    scaled = values / scale
    if np.ptp(scaled) == 0:
        raise ValueError("All selected values are identical; a continuous KDE is undefined.")
    return gaussian_kde(scaled)(grid / scale) / scale


def calculate_time_rmse(prediction, reference, reference_indices, mask):
    """RMSE across the same held-out samples at each time, in meters."""
    positions = np.flatnonzero(mask)
    rmse = np.zeros(prediction.shape[0])
    for start in range(0, len(positions), BATCH_SIZE):
        selected = positions[start:start + BATCH_SIZE]
        estimated = prediction[:, selected]
        actual = reference[:, reference_indices[selected]]
        # Scale before subtraction/squaring to avoid overflow on large finite values.
        scale = np.maximum(np.max(np.abs(estimated), axis=1),
                           np.max(np.abs(actual), axis=1))
        scale[scale == 0.0] = 1.0
        difference_scaled = estimated / scale[:, None] - actual / scale[:, None]
        with np.errstate(over="ignore"):
            contribution = scale * np.sqrt(
                np.sum(difference_scaled ** 2, axis=1) / len(positions)
            )
            rmse = np.hypot(rmse, contribution)
    return rmse


def plot_comparison(results, reference_path, training_indices):
    first = results[0]["population"]
    time = first["time"]
    indices = first["reference_indices"]
    time_indices = [int(np.argmin(np.abs(time - t))) for t in (5.0, 10.0)]
    if not np.allclose(time[time_indices], [5.0, 10.0]):
        raise ValueError("The reference time grid must contain t=5 and t=10.")

    with np.load(reference_path, allow_pickle=False) as data:
        if not np.array_equal(time, data["time"]):
            raise ValueError("RK4 and model time grids differ.")
        reference_inputs = data["samples"]
        reference = data["y2_all"]
    if not np.array_equal(first["samples"], reference_inputs[indices]):
        raise ValueError("Prediction inputs do not match the RK4 inputs.")

    reference_qoi = np.full((3, len(indices)), np.nan)
    reference_finite = np.zeros(len(indices), dtype=bool)
    for start in range(0, len(indices), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(indices))
        values = reference[:, indices[start:stop]]
        reference_finite[start:stop] = np.isfinite(values).all(axis=0)
        reference_qoi[:2, start:stop] = values[time_indices, :]
        reference_qoi[2, start:stop] = np.max(np.abs(values), axis=0)

    held_out = ~np.isin(indices, training_indices)
    common = held_out & reference_finite
    for result in results:
        population = result["population"]
        if (not np.array_equal(indices, population["reference_indices"])
                or not np.array_equal(first["samples"], population["samples"])
                or not np.array_equal(time, population["time"])):
            raise ValueError("All models must use the same prediction inputs and time grid.")
        result["qoi"], result["finite"] = extract_qoi(population["prediction"], time_indices)
        common &= result["finite"]
        print(f"{result['label']}: nonfinite trajectories "
              f"{np.count_nonzero(~result['finite']):,}/{len(indices):,}", flush=True)

    n_common = int(common.sum())
    if n_common < 2:
        raise RuntimeError("Fewer than two common finite held-out trajectories; cannot compare PDFs.")
    print(f"Common finite held-out trajectories: {n_common:,}/{held_out.sum():,}")
    for result in results:
        result["time_rmse"] = calculate_time_rmse(
            result["population"]["prediction"], reference, indices, common
        )
    del reference

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    titles = ("t = 5 s", "t = 10 s", "Maximum absolute displacement")
    # Color identifies the coefficient surrogate; line style identifies NARX selection.
    colors = {"kriging": "tab:blue", "pce_lars": "tab:green", "pdd": "tab:orange"}
    styles = {"lars": "--"}
    for q, ax in enumerate(axes):
        reference_values = reference_qoi[q, common]
        lower, upper = reference_values.min(), reference_values.max()
        padding = 0.05 * max(upper - lower, 1e-6)
        grid = np.linspace(lower - padding, upper + padding, 500)
        ax.plot(grid, evaluate_density(reference_values, grid),
                color="red", linewidth=2, label="RK4 MCS")
        outside_text = []
        for result in results:
            values = result["qoi"][q, common]
            outside = np.count_nonzero((values < grid[0]) | (values > grid[-1]))
            outside_text.append(f"{result['label']}: {outside:,}")
            try:
                density = evaluate_density(values, grid)
            except (ValueError, np.linalg.LinAlgError) as error:
                raise RuntimeError(f"KDE failed for {result['label']}, {titles[q]}: {error}") from error
            ax.plot(grid, density, color=colors[result["coefficient_method"]],
                    linestyle=styles[result["narx_method"]], linewidth=1.5,
                    label=result["label"])
        ax.set(title=titles[q], ylabel="Probability density (1/m)",
               xlabel="y2 (m)" if q < 2 else "max |y2(t)| (m)")
        ax.grid(alpha=0.25)
        ax.set_xlim(grid[0], grid[-1])
        ax.text(0.98, 0.97, "Outside displayed range:\n" + "\n".join(outside_text),
                transform=ax.transAxes, ha="right", va="top", fontsize=7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(f"Three LARS-selected NARX surrogate models vs RK4: y2 PDFs\n"
                 f"Common finite held-out trajectories: {n_common:,}/{held_out.sum():,}; "
                 "KDE uses all paired values; view limited to RK4 range", fontsize=12)
    invalid_text = "; ".join(
        f"{r['label']}: {np.count_nonzero(held_out & ~r['finite']):,}"
        for r in results
    )
    fig.text(0.5, 0.12, "Nonfinite held-out trajectories — " + invalid_text,
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.16, 1, 0.89))
    fig.savefig(OUTPUT_DIR / "three_models_pdf_comparison.png", dpi=180)
    fig.savefig(OUTPUT_DIR / "three_models_pdf_comparison.pdf")

    rmse_fig, rmse_ax = plt.subplots(figsize=(11, 6))
    for result in results:
        rmse = result["time_rmse"]
        rmse_ax.plot(time, rmse, color=colors[result["coefficient_method"]],
                     linewidth=1.5, label=result["label"])
        if not np.isfinite(rmse).all():
            print(f"{result['label']}: RMSE exceeds numerical range at "
                  f"{np.count_nonzero(~np.isfinite(rmse))} time points.", flush=True)
    rmse_ax.set(xlabel="Time (s)", ylabel="RMSE (m)",
                title="Time-dependent y2 RMSE against RK4")
    rmse_ax.grid(alpha=0.25)
    rmse_ax.legend()
    rmse_fig.suptitle(
        f"Common finite held-out trajectories: {n_common:,}/{held_out.sum():,}; "
        "statistics conditional on finite trajectories", fontsize=10
    )
    rmse_fig.text(0.5, 0.02, "Excluded nonfinite held-out trajectories — " + invalid_text,
                  ha="center", fontsize=8)
    rmse_fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    rmse_fig.savefig(OUTPUT_DIR / "three_models_time_rmse.png", dpi=180)
    rmse_fig.savefig(OUTPUT_DIR / "three_models_time_rmse.pdf")
    np.savez(
        OUTPUT_DIR / "three_models_qoi.npz", time=time,
        reference_indices=indices, training_indices=training_indices,
        labels=np.array([r["label"] for r in results]),
        reference_qoi=reference_qoi, model_qoi=np.stack([r["qoi"] for r in results]),
        model_finite=np.stack([r["finite"] for r in results]),
        held_out_mask=held_out, common_finite_mask=common,
        prediction_seed=PREDICTION_SEED,
        time_rmse=np.stack([r["time_rmse"] for r in results]),
        rmse_definition="sqrt(mean((prediction - RK4)^2)) across common finite held-out samples",
    )
    print(f"Saved comparison: {OUTPUT_DIR}", flush=True)
    plt.show()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pce_module = load_pce_module()
    results = []
    training_sets = []
    reference_path = None
    for narx_method in NARX_METHODS:
        module = lars_narx
        if reference_path is not None and module.DATA_PATH.resolve() != reference_path.resolve():
            raise ValueError("Both NARX methods must use the same RK4 reference file.")
        reference_path = module.DATA_PATH
        print(f"\nSelecting NARX structure: {narx_method}", flush=True)
        narx_result = module.main()
        if training_sets and not np.array_equal(training_sets[0], narx_result["training_indices"]):
            raise ValueError("Both NARX methods must use identical training indices for this comparison.")
        training_sets.append(narx_result["training_indices"])
        # Refit the LARS-selected structure once for all three coefficient models.
        a = kriging.fit_coefficient_matrix(narx_result, narx_method)
        for coefficient_method in COEFFICIENT_METHODS:
            label = f"{narx_method.upper()} + {coefficient_method.upper()}"
            print(f"\nTraining {label}", flush=True)
            predict = build_coefficient_predictor(
                coefficient_method, narx_result["samples_train"], a, pce_module
            )
            population = run_population_prediction(
                time=narx_result["time"], initial_state=narx_result["initial_state"],
                terms=narx_result["best_terms"], predict_coefficients=predict,
                n_samples=N_PREDICT, seed=PREDICTION_SEED, batch_size=BATCH_SIZE,
                max_lag=module.MAX_LAG,
                output_dir=OUTPUT_DIR / f"{narx_method}_{coefficient_method}",
                model_name=label, training_indices=narx_result["training_indices"],
                reference_path=reference_path,
            )
            results.append({"label": label, "narx_method": narx_method,
                            "coefficient_method": coefficient_method, "population": population})
    plot_comparison(results, reference_path, np.unique(np.concatenate(training_sets)))


if __name__ == "__main__":
    main()
