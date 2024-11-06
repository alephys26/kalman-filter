import numpy as np
from config import delta_t, R


def true_movement(state, acceleration):
    position = state[:3]
    velocity = state[3:]
    new_velocity = velocity + acceleration * delta_t
    new_position = position + velocity * delta_t
    return np.hstack((new_position, new_velocity))


def get_noisy_measurement(true_state):
    noise = np.random.multivariate_normal(np.zeros(6), R)
    return true_state + noise
