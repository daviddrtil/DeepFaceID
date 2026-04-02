import numpy as np

class PassiveScoreAggregator:
    def run(self, spatial, frequency, temporal):
        return float(np.clip(1.0 * spatial + 0.0 * frequency + 0.0 * temporal, 0.0, 1.0))
