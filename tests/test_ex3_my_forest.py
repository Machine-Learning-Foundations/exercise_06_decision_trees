import pytest
import numpy as np
from collections import Counter
from scipy.stats import mode
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from src.ex3_my_forest import entropy, information_gain, build_tree, RandomForest, DecisionNode, Leaf

@pytest.fixture
def dataset():
    # Create a fixture for the dataset
    dataset = load_wine()
    missing_rate = 0.05
    mask = np.random.rand(*dataset.data.shape) < missing_rate
    dataset.data[mask] = np.nan
    return dataset

@pytest.fixture
def train_test_data(dataset):
    # Create a fixture for train-test split data
    xtrain, xtest, ytrain, ytest = train_test_split(dataset.data, dataset.target, train_size=0.75, random_state=29)
    return xtrain, xtest, ytrain, ytest

@pytest.fixture
def random_forest(train_test_data):
    # Create a fixture for RandomForest instance
    xtrain, _, ytrain, _ = train_test_data
    rf = RandomForest(n_trees=10, max_depth=5, n_features=4)
    rf.fit(xtrain, ytrain)
    return rf

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


def test_build_tree():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])
    features = [0, 1]
    tree = build_tree(X, y, features, max_depth=2)
    
    assert isinstance(tree, DecisionNode)
    assert isinstance(tree.left, Leaf)
    assert isinstance(tree.right, DecisionNode)
    assert isinstance(tree.right.left, Leaf)
    assert isinstance(tree.right.right, Leaf)
    
    # Check the numerical threshold in the root decision node
    expected_threshold_root = 1
    assert np.isclose(tree.threshold, expected_threshold_root)
    
    # Check the numerical threshold in the left decision node
    expected_threshold_right = 3
    assert np.isclose(tree.right.threshold, expected_threshold_right)

    # Verify the predictions in the leaf nodes
    assert tree.right.left.predictions == Counter([1])



def test_random_forest_fit(random_forest):
    assert len(random_forest.trees) == random_forest.n_trees

def test_random_forest_predict(random_forest, train_test_data):
    _, xtest, _, ytest = train_test_data
    predictions = random_forest.predict(xtest)
    accuracy = np.mean(predictions == ytest)
    
    # Test accuracy is within a reasonable range
    assert accuracy >= 0.85

