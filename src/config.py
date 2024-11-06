import numpy as np

# Simulation parameters
delta_t = 0.001  # Time step in seconds (1 millisecond)
T = 0.1  # RADAR measurement interval in seconds (100 milliseconds)
num_steps = 50  # Number of steps to simulate

# Initial state (position and velocity vectors)
initial_position = np.array([0.0, 0.0, 0.0])
initial_velocity = np.array([0.0, 0.0, 0.0])

# State transition matrix (A)
A = np.eye(6)
A[0:3, 3:6] = delta_t * np.eye(3)

# Control matrix (B)
B = np.zeros((6, 3))
B[3:6, :] = delta_t * np.eye(3)

# Measurement matrix (H)
H = np.eye(6)

# Covariance matrices
P_initial = np.eye(6)  # Initial estimate covariance
Q = 0.01 * np.eye(6)  # Process noise covariance
R = 0.1 * np.eye(6)  # Measurement noise covariance
