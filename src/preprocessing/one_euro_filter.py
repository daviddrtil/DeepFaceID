import math
import numpy as np

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.1, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _get_alpha(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (1.0 + r)

    def predict(self, t, x):
        t = float(t)
        x = np.asarray(x, dtype=np.float64)

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        if t_e <= 1e-5:
            return self.x_prev

        # Filter the derivative (speed)
        a_d = self._get_alpha(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Filter the actual signal
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat).mean()
        a = self._get_alpha(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
