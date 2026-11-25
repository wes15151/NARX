"""Shared input sampling, short RK4 initialization and NARX population prediction."""
import importlib.util
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from threadpoolctl import threadpool_limits


def generate_prediction_samples(n_samples, seed):
    rng = np.random.default_rng(seed)
    physical = rng.normal(
        [2000.0, 2000.0, 20.0, 40.0, 600.0],
        [200.0, 200.0, 2.0, 4.0, 60.0],
        size=(n_samples, 5),
    )
    amplitude = rng.uniform(0.09, 0.11, size=n_samples)
    omega = rng.uniform(1.8 * np.pi, 2.2 * np.pi, size=n_samples)
    return np.column_stack((physical, amplitude, omega))


def load_rk4_step():
    path = Path(__file__).resolve().parent.parent / "PCE_NARX" / "generate_quarter_car_data.py"
    spec = importlib.util.spec_from_file_location("quarter_car_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rk4_step


def initial_y2_history(inputs, time, initial_state, max_lag, rk4_step):
    # Existing RK4 arithmetic also accepts state (4, batch) and parameters (7, batch).
    state = np.broadcast_to(np.asarray(initial_state)[:, None], (4, len(inputs))).copy()
    history = np.empty((max_lag, len(inputs)))
    history[0] = state[0]
    for k in range(max_lag - 1):
        state = rk4_step(time[k], state, inputs.T, time[k + 1] - time[k])
        history[k + 1] = state[0]
    return history


def recursive_batch(inputs, time, initial_history, terms, coefficients, max_lag):
    prediction = np.full((len(time), len(inputs)), np.nan)
    prediction[:max_lag] = initial_history
    failed = ~np.all(np.isfinite(initial_history), axis=0)
    failed |= ~np.all(np.isfinite(coefficients), axis=1)

    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(max_lag, len(time)):
            next_value = np.zeros(len(inputs))
            for q, (l, m, j, k) in enumerate(terms):
                value = np.ones(len(inputs))
                if l:
                    value *= prediction[n - j] ** l
                if m:
                    road = inputs[:, 5] * np.sin(inputs[:, 6] * time[n - k])
                    value *= road ** m
                next_value += coefficients[:, q] * value
            failed |= ~np.isfinite(next_value)
            next_value[failed] = np.nan
            prediction[n] = next_value
            if failed.all():
                break
    return prediction, failed


@threadpool_limits.wrap(limits=1, user_api="blas")
def run_population_prediction(time, initial_state, terms, predict_coefficients,
                              n_samples, seed, batch_size, max_lag, output_dir,
                              model_name, training_indices, reference_path=None):
    if n_samples <= 0 or batch_size <= 0:
        raise ValueError("Prediction count and batch size must be positive.")
    if max_lag < 1 or len(time) <= max_lag:
        raise ValueError("The time grid must extend beyond the initial history.")
    if len(terms) == 0 or np.max(terms[:, 2:4]) > max_lag:
        raise ValueError("Invalid NARX terms or insufficient initialization lag.")

    started = perf_counter()
    reference_indices = None
    if reference_path is None:
        inputs = generate_prediction_samples(n_samples, seed)
        sample_source = "new random inputs"
    else:
        with np.load(reference_path, allow_pickle=False) as data:
            reference_inputs = data["samples"]
            if not np.array_equal(time, data["time"]):
                raise ValueError("Prediction and RK4 reference time grids differ.")
            if not np.array_equal(initial_state, data["initial_state"]):
                raise ValueError("Prediction and RK4 reference initial states differ.")
        if n_samples > len(reference_inputs):
            raise ValueError("Not enough stored RK4 inputs for the requested paired comparison.")
        reference_indices = np.random.default_rng(seed).choice(
            len(reference_inputs), size=n_samples, replace=False
        )
        inputs = reference_inputs[reference_indices]
        sample_source = "stored RK4 inputs, sampled without replacement"
        print(f"{model_name}: using {n_samples:,} stored inputs for paired RK4 comparison.",
              flush=True)
    rk4_step = load_rk4_step()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "y2_predictions.npy"
    metadata_path = output_dir / "prediction_metadata.npz"
    # Column-major storage makes writes of complete trajectory batches contiguous.
    trajectories = np.lib.format.open_memmap(
        trajectory_path, mode="w+", dtype=np.float64,
        shape=(len(time), n_samples), fortran_order=True,
    )
    np.savez(metadata_path, status="running", model_name=model_name,
             n_samples=n_samples, prediction_seed=seed)
    failed = np.zeros(n_samples, dtype=bool)
    coefficients_all = np.empty((n_samples, len(terms)))

    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        batch = inputs[start:stop]
        coefficients = np.asarray(predict_coefficients(batch))
        if coefficients.shape != (len(batch), len(terms)):
            raise ValueError("Predicted coefficients must have shape (batch size, NARX terms).")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            history = initial_y2_history(batch, time, initial_state, max_lag, rk4_step)
        values, invalid = recursive_batch(batch, time, history, terms, coefficients, max_lag)
        trajectories[:, start:stop] = values
        failed[start:stop] = invalid
        coefficients_all[start:stop] = coefficients
        print(f"{model_name}: predicted {stop:,}/{n_samples:,}; "
              f"numerically invalid={failed[:stop].sum():,}", flush=True)

    trajectories.flush()
    seconds = perf_counter() - started
    np.savez(metadata_path, status="complete", model_name=model_name,
             time=time, samples=inputs, narx_terms=terms,
             narx_coefficients=coefficients_all, failed_predictions=failed,
             training_indices=training_indices, prediction_seed=seed,
             initial_state=initial_state, max_lag=max_lag,
             initialization="short RK4, upper displacement state column 0",
             trajectory_file=trajectory_path.name, trajectory_axes="time, sample",
             sample_source=sample_source,
             reference_indices=(reference_indices if reference_indices is not None
                                else np.array([], dtype=int)),
             reference_path="" if reference_path is None else str(reference_path),
             prediction_seconds=seconds)
    print(f"{model_name}: total prediction time={seconds:.3f} s; "
          f"numerically invalid={failed.sum():,}/{n_samples:,}", flush=True)
    print(f"Saved trajectories: {trajectory_path}", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)
    return {"samples": inputs, "prediction": trajectories,
            "failed": failed, "narx_coefficients": coefficients_all,
            "trajectory_path": trajectory_path, "metadata_path": metadata_path,
            "prediction_seconds": seconds, "time": time,
            "reference_indices": reference_indices, "output_dir": output_dir}


def plot_population_comparison(population_result, reference_path, training_indices,
                              model_name, batch_size=2000):
    time = population_result["time"]
    prediction = population_result["prediction"]
    reference_indices = population_result["reference_indices"]
    if reference_indices is None:
        raise ValueError("Pointwise errors require prediction at stored RK4 inputs.")
    with np.load(reference_path, allow_pickle=False) as data:
        if not np.array_equal(time, data["time"]):
            raise ValueError("RK4 and prediction time grids differ.")
        if not np.array_equal(population_result["samples"], data["samples"][reference_indices]):
            raise ValueError("Prediction inputs do not match the paired RK4 inputs.")
        reference = data["y2_all"]

    time_indices = []
    for target in (5.0, 10.0):
        index = int(np.argmin(np.abs(time - target)))
        if not np.isclose(time[index], target):
            raise ValueError(f"Time {target:g} s is not on the stored time grid.")
        time_indices.append(index)

    held_out = ~np.isin(reference_indices, training_indices)
    paired = held_out & ~population_result["failed"]
    n_population = prediction.shape[1]
    error_scale = np.zeros(len(time))
    qoi_reference = np.full((3, n_population), np.nan)
    qoi_prediction = np.full((3, n_population), np.nan)

    # Work in batches rather than allocating a full time-by-population error matrix.
    for start in range(0, n_population, batch_size):
        stop = min(start + batch_size, n_population)
        positions = np.flatnonzero(paired[start:stop]) + start
        if not len(positions):
            continue
        actual = reference[:, reference_indices[positions]]
        estimated = prediction[:, positions]
        with np.errstate(over="ignore", invalid="ignore"):
            difference = estimated - actual
        valid = np.all(np.isfinite(actual) & np.isfinite(estimated) & np.isfinite(difference), axis=0)
        paired[positions[~valid]] = False
        positions = positions[valid]
        if not len(positions):
            continue
        actual, estimated, difference = actual[:, valid], estimated[:, valid], difference[:, valid]
        error_scale = np.maximum(error_scale, np.max(np.abs(difference), axis=1))
        qoi_reference[:, positions] = np.vstack((actual[time_indices], np.max(np.abs(actual), axis=0)))
        qoi_prediction[:, positions] = np.vstack((estimated[time_indices], np.max(np.abs(estimated), axis=0)))

    n_valid = int(paired.sum())
    if n_valid < 2:
        raise RuntimeError("Fewer than two finite held-out trajectories; comparison plots unavailable.")

    # Normalize by a common time-dependent scale to reduce overflow in moment sums.
    safe_scale = np.where(error_scale > 0, error_scale, 1.0)
    normalized_sum = np.zeros(len(time))
    normalized_square_sum = np.zeros(len(time))
    for start in range(0, n_population, batch_size):
        stop = min(start + batch_size, n_population)
        positions = np.flatnonzero(paired[start:stop]) + start
        if not len(positions):
            continue
        difference = prediction[:, positions] - reference[:, reference_indices[positions]]
        normalized = difference / safe_scale[:, None]
        normalized_sum += normalized.sum(axis=1)
        normalized_square_sum += np.sum(normalized ** 2, axis=1)
    normalized_mean = normalized_sum / n_valid
    normalized_variance = np.maximum(normalized_square_sum / n_valid - normalized_mean ** 2, 0.0)
    error_mean = safe_scale * normalized_mean
    with np.errstate(over="ignore", invalid="ignore"):
        error_variance = safe_scale * (safe_scale * normalized_variance)

    output_dir = population_result["output_dir"]
    excluded = int(held_out.sum()) - n_valid
    caption = (f"{model_name}: finite held-out trajectories {n_valid:,}/{held_out.sum():,}; "
               f"excluded {excluded:,}")
    print(caption, flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for q, (ax, title, xlabel) in enumerate(zip(
        axes, ("y2 at t=5 s", "y2 at t=10 s", "Maximum absolute y2 displacement"),
        ("y2 (m)", "y2 (m)", "max |y2(t)| (m)"),
    )):
        actual, estimated = qoi_reference[q, paired], qoi_prediction[q, paired]
        span = max(float(np.ptp(actual)), 1e-10)
        lower, upper = actual.min() - .1 * span, actual.max() + .1 * span
        if q == 2:
            lower = max(0., lower)
        grid = np.linspace(lower, upper, 600)
        for values, style, label in ((actual, "r-", "RK4 MCS"), (estimated, "b--", model_name)):
            if np.ptp(values) == 0:
                print(f"{title}: {label} has zero variance; KDE not defined.", flush=True)
                continue
            try:
                # Fit to all paired finite values, including values outside the view.
                density = gaussian_kde(values, bw_method="scott")(grid)
            except (np.linalg.LinAlgError, ValueError) as error:
                print(f"{title}: KDE failed for {label}: {error}", flush=True)
                continue
            ax.plot(grid, density, style, linewidth=1.5, label=label)
        outside = int(np.sum((estimated < lower) | (estimated > upper)))
        ax.text(.98, .72, f"Model: {outside} outside view", transform=ax.transAxes,
                ha="right", fontsize=8)
        ax.set(title=title, xlabel=xlabel, ylabel="Probability density (1/m)", xlim=(lower, upper))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle(caption)
    fig.tight_layout()
    fig.savefig(output_dir / "pdf_comparison.png", dpi=180)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, values, title, ylabel in (
        (axes[0], error_mean, "Mean of signed prediction error", "Mean error (m)"),
        (axes[1], error_variance, "Variance of signed prediction error (ddof=0)", "Error variance (m^2)"),
    ):
        finite = np.isfinite(values)
        ax.plot(time, np.where(finite, values, np.nan), "b-", linewidth=1.3)
        ax.set(title=title, ylabel=ylabel)
        ax.grid(alpha=.25)
        if not finite.all():
            ax.text(.02, .95, f"{np.sum(~finite)} time points overflowed; omitted from line",
                    transform=ax.transAxes, va="top", fontsize=9)
    axes[0].axhline(0, color="black", linewidth=.7, linestyle="--")
    axes[1].set_xlabel("Time (s)")
    fig.suptitle(caption + "\nError = prediction - RK4; statistics conditional on finite trajectories")
    fig.tight_layout()
    fig.savefig(output_dir / "error_mean_variance.png", dpi=180)
    np.savez(output_dir / "error_statistics.npz", time=time, error_mean=error_mean,
             error_variance=error_variance, variance_ddof=0, paired_mask=paired,
             held_out_mask=held_out, reference_indices=reference_indices,
             error_definition="prediction minus RK4", n_valid=n_valid,
             excluded_invalid=excluded)
    plt.show()
    return {"error_mean": error_mean, "error_variance": error_variance, "paired_mask": paired}
