"""Paired comparison of LHS and two adaptive rules on fresh training seeds."""
import matplotlib.pyplot as plt
import numpy as np

import tune_kriging_adaptive as tuning
from quarter_car_prediction import initial_y2_history, load_rk4_step


SEEDS = range(31, 36)
LABELS = ["LHS 100", "Adaptive tuned", "Adaptive untuned"]
OUTPUT = tuning.OUTPUT / "tuned_vs_untuned"


def main():
    with np.load(tuning.OUTPUT / "final_rmse.npz") as saved:
        tuned = saved["exponents"]
        test = saved["test_indices"]
    model = tuning.model
    with np.load(model.narx.DATA_PATH) as f:
        data = (f["samples"], f["y2_all"], f["time"], f["initial_state"])
    inputs, _, time, state = data
    order = np.random.default_rng(2026).permutation(len(inputs))
    pool_ids = order[:2 * model.CANDIDATE_POOL_SIZE]
    if not np.array_equal(order[len(pool_ids) + 2000:], test):
        raise ValueError("Stored evaluation split differs from the tuning experiment.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    curves = []
    print(f"Tuned: {tuned}; untuned: {tuning.DEFAULT}", flush=True)
    for seed in SEEDS:
        print(f"Preparing seed {seed}", flush=True)
        design = tuning.prepare(seed, data, pool_ids)
        x = model.lhs_inputs(model.N_FINAL, design["lhs_rng"])
        y = initial_y2_history(x, time, state, len(time), load_rk4_step())
        terms, _, gp = model.train_baseline(x, y, time, np.arange(len(x)))
        seed_curves = [tuning.error_curve(gp, terms, data, test)]
        for label, exponents in zip(LABELS[1:], (tuned, tuning.DEFAULT)):
            print(f"Seed {seed}: {label}", flush=True)
            gp = tuning.enrich(design, exponents, data)
            seed_curves.append(tuning.error_curve(gp, design["terms"], data, test))
        curves.append(seed_curves)
        np.savez(OUTPUT / "seed_rmse.npz", seeds=list(SEEDS)[:len(curves)],
                 labels=LABELS, time=time, curves=curves, tuned=tuned,
                 untuned=tuning.DEFAULT, test_indices=test)

    curves = np.asarray(curves)
    finite = np.isfinite(curves).all(axis=2)
    keep = finite.all(axis=1)
    for i, label in enumerate(LABELS):
        print(f"{label} nonfinite seeds: {np.asarray(SEEDS)[~finite[:, i]].tolist()}")
    if not keep.any():
        raise RuntimeError("No common finite seed. See seed_rmse.npz for individual curves.")
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for label, curve in zip(LABELS, curves[keep].mean(axis=0)):
        ax.plot(time, curve, label=label)
        print(f"{label}: time-averaged RMSE = {curve.mean():.6g} m")
    ax.set(xlabel="Time (s)", ylabel="Mean y2 RMSE (m)",
           title=f"RK4 comparison on {keep.sum()}/{len(SEEDS)} common finite seeds")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(OUTPUT / "mean_time_rmse.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
