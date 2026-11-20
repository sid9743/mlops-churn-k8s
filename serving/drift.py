"""Population Stability Index (PSI) on a rolling window of live requests.

PSI compares the live distribution of a feature with the training distribution:
below 0.1 is stable, 0.1 to 0.25 is a moderate shift, above 0.25 means the model
should be retrained.
"""
from collections import deque

import numpy as np


class PsiMonitor:
    def __init__(self, bin_edges: list[float], window: int = 500):
        self.edges = np.array(bin_edges)
        self.expected = np.full(len(bin_edges) - 1, 1 / (len(bin_edges) - 1))
        self.values: deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self.values.append(value)

    def psi(self) -> float | None:
        if len(self.values) < 50:
            return None
        clipped = np.clip(np.array(self.values), self.edges[0], self.edges[-1])
        counts, _ = np.histogram(clipped, bins=self.edges)
        actual = np.clip(counts / counts.sum(), 1e-4, None)
        expected = np.clip(self.expected, 1e-4, None)
        return float(np.sum((actual - expected) * np.log(actual / expected)))
