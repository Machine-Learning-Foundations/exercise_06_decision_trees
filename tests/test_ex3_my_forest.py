import pytest
import numpy as np

from src.ex3_my_forest import entropy, information_gain


def test_entropy():
    y = np.array([0, 0, 1, 1, 1, 1])
    calculated_entropy = entropy(y)
    expected_entropy = 0.9182958340544896  # Manually calculated
    assert np.isclose(calculated_entropy, expected_entropy, atol=1e-5)

def test_information_gain():
    y = np.array([0, 0, 1, 1, 1, 1])
    y_left = np.array([0, 0])
    y_right = np.array([1, 1, 1, 1])
    calculated_gain = information_gain(y, y_left, y_right)
    expected_gain = 0.9182958340544896  # Entropy(y) - 0*Entropy(y_left) - 1*Entropy(y_right)
    assert np.isclose(calculated_gain, expected_gain, atol=1e-5)
