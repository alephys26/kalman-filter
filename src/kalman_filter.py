import numpy as np
from config import A, B, H, Q, R, P_initial


class KalmanFilter:
    def __init__(self, initial_state):
        self.state = initial_state
        self.P = P_initial

    def predict(self, acceleration):
        self.state = A @ self.state + B @ acceleration
        self.P = A @ self.P @ A.T + Q
        return self.state

    def update(self, measurement):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ (measurement - H @ self.state)
        self.P = (np.eye(6) - K @ H) @ self.P
        return self.state
