from time import perf_counter
from pathlib import Path
import numpy as np

def quarter_car_rhs(t, state, parameters):
    ks, ku, ms, mu, c, amplitude, omega = parameters
    xs, vs, xu, vu = state
    road = amplitude * np.sin(omega * t)
    suspension_force = ks * (xs - xu) ** 3 + c * (vs - vu)
    acceleration_s = -suspension_force / ms
    acceleration_u = (suspension_force + ku * (road - xu)) / mu
    return np.array([vs, acceleration_s, vu, acceleration_u])


def rk4_step(t, state, parameters, dt):
    k1 = quarter_car_rhs(t, state, parameters)
    k2 = quarter_car_rhs(t + dt / 2, state + dt * k1 / 2, parameters)
    k3 = quarter_car_rhs(t + dt / 2, state + dt * k2 / 2, parameters)
    k4 = quarter_car_rhs(t + dt, state + dt * k3, parameters)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def simulate_quarter_car(parameters, time, initial_state):
    states = np.zeros((len(time), 4))
    states[0] = initial_state
    for k in range(len(time) - 1):
        step_size = time[k + 1] - time[k]
        states[k + 1] = rk4_step(time[k], states[k], parameters, step_size)
    return states



def main():
    script_started = perf_counter()
    n_samples = 100000
    dt = 0.01
    t_end = 10.0
    n_steps = int(round(t_end / dt))
    time = np.linspace(0.0, t_end, n_steps + 1)
    rng = np.random.default_rng(42)

    normal_mean = np.array([2000.0, 2000.0, 20.0, 40.0, 600.0])
    normal_std = np.array([200.0, 200.0, 2.0, 4.0, 60.0])
    amplitude_bounds = (0.09, 0.11)
    omega_bounds = (1.8 * np.pi, 2.2 * np.pi)

    initial_state = np.zeros(4)

    physical_samples = rng.normal(normal_mean, normal_std, size=(n_samples, 5))
    amplitude_samples = rng.uniform(*amplitude_bounds, size=n_samples)
    omega_samples = rng.uniform(*omega_bounds, size=n_samples)
    samples = np.column_stack((physical_samples, amplitude_samples, omega_samples))


    y1_all = np.zeros((len(time), n_samples))
    y2_all = np.zeros((len(time), n_samples))
    # %%
    rk4_started = perf_counter()
    for i in range(n_samples):
        states = simulate_quarter_car(samples[i], time, initial_state)
        y1_all[:, i] = states[:, 2]
        y2_all[:, i] = states[:, 0]
        if (i + 1) % 100 == 0:
            print(f'Completed {i + 1}/{n_samples}', flush=True)
    rk4_seconds = perf_counter() - rk4_started
    print(f"RK4 {n_samples} samples: {rk4_seconds:.3f} s", flush=True)
    # %%
    train_indices = rng.choice(n_samples, size=100, replace=False)
    data_path = Path(__file__).resolve().parent / "quarter_car_data.npz"
    np.savez(data_path, time=time, samples=samples,
             y1_all=y1_all, y2_all=y2_all,
             train_indices=train_indices, dt=dt, initial_state=initial_state,
             rk4_seconds=rk4_seconds,
             y1_state_index=2, y2_state_index=0)
    print(f"Saved data: {data_path}")
    print(f"Total data-generation script time: {perf_counter() - script_started:.3f} s")


if __name__ == "__main__":
    main()
