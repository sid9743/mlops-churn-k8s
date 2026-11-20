import numpy as np

from serving.drift import PsiMonitor

EDGES = np.linspace(20, 150, 11).tolist()


def test_needs_minimum_samples():
    m = PsiMonitor(EDGES)
    for v in range(10):
        m.add(50 + v)
    assert m.psi() is None


def test_same_distribution_is_stable():
    m = PsiMonitor(EDGES)
    for v in np.random.default_rng(0).uniform(20, 150, 500):
        m.add(v)
    assert m.psi() < 0.1


def test_shifted_distribution_is_flagged():
    m = PsiMonitor(EDGES)
    for v in np.random.default_rng(0).uniform(120, 150, 500):
        m.add(v)
    assert m.psi() > 0.25
