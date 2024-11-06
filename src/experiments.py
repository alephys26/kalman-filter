import numpy as np
import os
from config import initial_position, initial_velocity, delta_t, num_steps
from kalman_filter import KalmanFilter
from simulation import true_movement, get_noisy_measurement
from visualization import plot_positions, plot_errors


def get_acceleration_pattern(pattern, step):
    if pattern == "constant":
        return np.array([0.1, 0.0, 0.05])

    elif pattern == "time_dependent":
        if step < num_steps / 2:
            return np.array([0.1, 0.0, 0.05])
        else:
            return np.array([0.0, -0.1, -0.05])

    elif pattern == "sinusoidal":
        return np.array([
            0.1 * np.sin(0.2 * step),
            0.1 * np.cos(0.2 * step),
            0.05 * np.sin(0.1 * step)
        ])

    elif pattern == "exponential_increase":
        return np.array([
            0.1 * np.exp(0.02 * step),
            0.0,
            0.05 * np.exp(0.015 * step)
        ])

    elif pattern == "exponential_decrease":
        return np.array([
            0.1 * np.exp(-0.02 * step),
            0.0,
            0.05 * np.exp(-0.015 * step)
        ])

    elif pattern == "random":
        return np.random.uniform(-0.1, 0.1, 3)

    elif pattern == "piecewise":
        if step < num_steps / 3:
            return np.array([0.1, 0.0, 0.05])
        elif step < 2 * num_steps / 3:
            return np.array([
                0.1 * np.sin(0.2 * step),
                0.1 * np.cos(0.2 * step),
                0.05 * np.sin(0.1 * step)
            ])
        else:
            return np.array([0.0, -0.1, -0.05])

    else:
        return np.array([0.0, 0.0, 0.0])


def save_experiment_output(experiment_name, true_positions, estimated_positions, errors):
    output_dir = f"../results/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)

    np.savetxt(f"{output_dir}/true_positions.csv",
               true_positions, delimiter=",")
    np.savetxt(f"{output_dir}/estimated_positions.csv",
               estimated_positions, delimiter=",")
    np.savetxt(f"{output_dir}/errors.csv", errors, delimiter=",")

    with open(f"{output_dir}/experiment_summary.txt", "w") as f:
        f.write(f"Experiment Summary for {experiment_name}:\n")
        f.write(f"Number of steps: {len(errors)}\n")
        f.write(f"Final error: {errors[-1]}\n")


def run_experiment(T, acceleration_pattern, measurement_noise_cov, experiment_name):
    initial_state = np.hstack((initial_position, initial_velocity))
    kalman_filter = KalmanFilter(initial_state)

    global R, Q
    R = measurement_noise_cov
    Q = R

    true_states = [initial_state]
    estimated_states = [initial_state]
    errors = []

    for step in range(1, num_steps + 1):
        acceleration = get_acceleration_pattern(
            acceleration_pattern, step)

        true_state = true_movement(true_states[-1], acceleration)
        true_states.append(true_state)

        predicted_state = kalman_filter.predict(acceleration)

        if step % int(T / delta_t) == 0:
            measurement = get_noisy_measurement(true_state)
            estimated_state = kalman_filter.update(measurement)
        else:
            estimated_state = predicted_state

        estimated_states.append(estimated_state)
        error = np.linalg.norm(true_state[:3] - estimated_state[:3])
        errors.append(error)

    true_positions = np.array([state[:3] for state in true_states])
    estimated_positions = np.array([state[:3] for state in estimated_states])

    save_experiment_output(experiment_name, true_positions,
                           estimated_positions, errors)

    plot_positions(true_positions, estimated_positions, experiment_name)
    plot_errors(errors, experiment_name)


T_values = [0.1, 0.2, 0.3]

acceleration_patterns = [
    "constant",
    "sinusoidal",
    "exponential_increase",
    "exponential_decrease",
    "random",
    "piecewise",
    "variable"
]

covariance_matrices = [
    0.1 * np.eye(6),
    0.15 * np.eye(6),
    0.2 * np.eye(6)
]

experiments = []

for T in T_values:
    for acceleration_pattern in acceleration_patterns:
        for measurement_noise_cov in covariance_matrices:
            experiment = {
                "T": T,
                "acceleration_pattern": acceleration_pattern,
                "measurement_noise_cov": measurement_noise_cov
            }
            experiments.append(experiment)
