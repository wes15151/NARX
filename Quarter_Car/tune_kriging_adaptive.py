"""Bayesian optimization of adaptive sampling exponents on fixed RK4 validation data."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

import compare_kriging_adaptive as model
from quarter_car_prediction import initial_y2_history, load_rk4_step, recursive_batch


TUNING_SEEDS = range(1, 6)
FINAL_SEEDS = range(11, 31)
N_TRIALS = 20
BO_SEED = 42
# Coordinates are PDF exponent alpha, variance exponent beta, sensitivity exponent gamma.
LOW = np.array([0., 0.5, 0.])
HIGH = np.array([1., 2., 2.])
DEFAULT = np.array([0.5, 1., 1.])
OUTPUT = Path(__file__).resolve().parent / "results" / "kriging_exponent_tuning"


def prepare(seed, data, pool_ids):
    inputs, reference, time, state = data
    initial_rng, lhs_rng, pool_rng, random_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    x = model.lhs_inputs(model.N_INITIAL, initial_rng)
    y = initial_y2_history(x, time, state, len(time), load_rk4_step())
    terms, a, gp = model.train_baseline(x, y, time, np.arange(len(x)))
    pool = pool_rng.choice(pool_ids, model.CANDIDATE_POOL_SIZE, replace=False)
    return dict(x=x, y=y, terms=terms, a=a, gp=gp, pool=pool, cache={},
                lhs_rng=lhs_rng, random_rng=random_rng)


def fit_added(design, ids, data):
    inputs, reference, time, _ = data
    cache = design["cache"]
    for i in ids:
        if i not in cache:
            cache[i] = model.fit_one_trajectory(i, inputs, reference, time, design["terms"])
    coefficients = np.column_stack([design["a"], *(cache[i] for i in ids)])
    return model.fit_gp(np.vstack((design["x"], inputs[ids])), coefficients,
                        design["gp"]["input_mean"], design["gp"]["input_std"])


def enrich(design, exponents, data):
    inputs, _, time, _ = data
    gp = design["gp"]
    pool = design["pool"].copy()
    ids = []
    for step in range(model.N_FINAL - model.N_INITIAL):
        if step % model.SENSITIVITY_UPDATE_INTERVAL == 0:
            weights = model.recursive_sensitivity_weights(
                design["x"], time, design["terms"], model.predict_coefficients(gp, design["x"]),
                design["y"][:model.narx.MAX_LAG])
        scores, _, _ = model.acquisition_scores(gp, inputs[pool], weights, *exponents)
        pos = int(np.argmax(scores))
        ids.append(pool[pos])
        pool = np.delete(pool, pos)
        gp = fit_added(design, ids, data)
    return gp


def error_curve(gp, terms, data, ids):
    inputs, reference, time, _ = data
    norm = np.zeros(len(time))
    lag = model.narx.MAX_LAG
    for start in range(0, len(ids), model.BATCH_SIZE):
        batch = ids[start:start + model.BATCH_SIZE]
        truth = reference[:, batch]
        coeff = model.predict_coefficients(gp, inputs[batch])
        pred, _ = recursive_batch(inputs[batch], time, truth[:lag], terms, coeff, lag)
        residual = pred - truth
        residual[~np.isfinite(residual)] = np.inf
        norm = np.hypot(norm, np.hypot.reduce(residual, axis=1))
    return norm / np.sqrt(len(ids))


def propose(rng, tried, scores):
    if not tried:
        return DEFAULT.copy()
    candidates = rng.random((2048, 3))
    if len(tried) < 5 or not np.any(np.isfinite(scores)):
        unit = candidates[0]
    else:
        losses = np.array(scores)
        valid = losses[np.isfinite(losses)]
        penalty = valid.max() + max(valid.max(), 1e-9)
        losses = np.where(np.isfinite(losses), losses, penalty)
        gp = GaussianProcessRegressor(kernel=Matern(np.ones(3), nu=2.5),
                                      normalize_y=True, alpha=1e-6)
        gp.fit((np.asarray(tried) - LOW) / (HIGH - LOW), losses)
        mean, std = gp.predict(candidates, return_std=True)
        gain = valid.min() - mean
        z = gain / np.maximum(std, 1e-15)
        improvement = gain * ndtr(z) + std * np.exp(-0.5*z*z) / np.sqrt(2*np.pi)
        unit = candidates[np.argmax(improvement)]
    return LOW + unit * (HIGH - LOW)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(model.narx.DATA_PATH) as f:
        data = (f["samples"], f["y2_all"], f["time"], f["initial_state"])
    inputs, _, time, state = data
    order = np.random.default_rng(2026).permutation(len(inputs))
    pool_ids = order[:2 * model.CANDIDATE_POOL_SIZE]
    validation = order[len(pool_ids):len(pool_ids) + 2000]
    test = order[len(pool_ids) + 2000:]
    if not len(test):
        raise ValueError("RK4 population too small for independent tuning and final evaluation.")
    designs = [prepare(seed, data, pool_ids) for seed in TUNING_SEEDS]
    rng = np.random.default_rng(BO_SEED)
    tried, scores = [], []
    for trial in range(N_TRIALS):
        params = propose(rng, tried, scores)
        errors = []
        for seed, design in zip(TUNING_SEEDS, designs):
            print(f"Trial {trial + 1}/{N_TRIALS}, seed {seed}, exponents {params}", flush=True)
            gp = enrich(design, params, data)
            errors.append(error_curve(gp, design["terms"], data, validation).mean())
        tried.append(params)
        scores.append(np.mean(errors))
        print(f"Mean RMSE {scores[-1]:.6g} m", flush=True)
        np.savetxt(OUTPUT / "trials.csv", np.column_stack((tried, scores,
                   np.minimum.accumulate(scores))), delimiter=",",
                   header="alpha,beta,gamma,mean_rmse,best_so_far", comments="")
    if not np.isfinite(scores).any():
        raise RuntimeError("No trial produced finite validation RMSE on all tuning seeds.")
    best = tried[np.argmin(scores)]
    print(f"Selected alpha, beta, gamma: {best}", flush=True)
    fig, ax = plt.subplots()
    trials = np.arange(1, N_TRIALS + 1)
    ax.plot(trials, scores, "o-", label="Trial mean RMSE")
    ax.plot(trials, np.minimum.accumulate(scores), "--", label="Best so far")
    ax.set(xlabel="Optimization trial", ylabel="Mean validation RMSE (m)")
    ax.legend()
    fig.savefig(OUTPUT / "tuning_rmse.png", bbox_inches="tight")

    curves = []
    labels = ["LHS 100", "LHS 50 + random 50", "LHS 50 + tuned adaptive 50"]
    for seed in FINAL_SEEDS:
        print(f"Final evaluation, seed {seed}", flush=True)
        design = prepare(seed, data, pool_ids)
        x = model.lhs_inputs(model.N_FINAL, design["lhs_rng"])
        y = initial_y2_history(x, time, state, len(time), load_rk4_step())
        lhs_terms, _, lhs_gp = model.train_baseline(x, y, time, np.arange(len(x)))
        ids = design["random_rng"].choice(design["pool"], model.N_FINAL-model.N_INITIAL,
                                          replace=False)
        fits = [(lhs_gp, lhs_terms), (fit_added(design, ids, data), design["terms"]),
                (enrich(design, best, data), design["terms"])]
        curves.append([error_curve(gp, terms, data, test) for gp, terms in fits])
        np.savez(OUTPUT / "final_rmse.npz", time=time, curves=curves, labels=labels,
                 seeds=list(FINAL_SEEDS)[:len(curves)], exponents=best, test_indices=test)
    fig, ax = plt.subplots()
    for label, curve in zip(labels, np.mean(curves, axis=0)):
        ax.plot(time, curve, label=label)
    ax.set(xlabel="Time (s)", ylabel="Mean y2 RMSE (m)",
           title=f"RK4 comparison across {len(FINAL_SEEDS)} final seeds")
    ax.legend()
    fig.savefig(OUTPUT / "final_time_rmse.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
