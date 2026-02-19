import math
import numpy as np

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        min_cutoff: Lower values = less jitter when slow. (Try 0.1 to 1.0)
        beta: Higher values = less lag when moving fast. (Try 0.01 to 0.1)
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = x0
        self.dx_prev = np.zeros_like(x0)
        self.t_prev = t0

    def alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t, x):
        dt = t - self.t_prev
        # Estimate the current variation (velocity)
        d_alpha = self.alpha(self.d_cutoff, dt)
        dx = (x - self.x_prev) / dt
        edx = d_alpha * dx + (1 - d_alpha) * self.dx_prev

        # Use the velocity to adapt the cutoff frequency
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        alpha = self.alpha(cutoff, dt)
        
        # Filter the signal
        x_hat = alpha * x + (1 - alpha) * self.x_prev

        # Update state
        self.x_prev = x_hat
        self.dx_prev = edx
        self.t_prev = t
        return x_hat