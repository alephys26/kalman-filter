import numpy as np
from config import initial_position, initial_velocity, num_steps, delta_t, T
from kalman_filter import KalmanFilter
from simulation import true_movement, get_noisy_measurement
from visualization import plot_positions, plot_errors

initial_state = np.hstack((initial_position, initial_velocity))
kalman_filter = KalmanFilter(initial_state)

true_states = [initial_state]
estimated_states = [initial_state]
errors = []

for step in range(1, num_steps + 1):
    if step < num_steps / 2:
        acceleration = np.array([0.1, 0.0, 0.0])
    else:
        acceleration = np.array([0.0, -0.1, 0.0])

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

plot_positions(true_positions, estimated_positions)
plot_errors(errors)
