import numpy as np
from collections import Counter
from scipy.stats import mode
from sklearn import ensemble
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

class Leaf:
    """
    Leaf node of the decision tree, representing a classification.

    Attributes:
    predictions (Counter): Count of labels at the leaf.
    """
    def __init__(self, y):
        self.predictions = Counter(y)

class DecisionNode:
    """
    Decision node of the decision tree, representing a split.

    Attributes:
    feature_index (int): Index of the feature used for the split.
    threshold (float): Threshold value for the split.
    left (DecisionNode or Leaf): Left child node.
    right (DecisionNode or Leaf): Right child node.
    """
    def __init__(self, feature_index, threshold, left, right):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

def entropy(y):
    """
    Calculate the entropy of a label array.

    Parameters:
    y (array-like): Array of labels.

    Returns:
    float: Entropy of the labels.
    """
    # 1) TODO: Implement me
    return None

def information_gain(y, y_left, y_right):
    """
    Calculate the information gain of a split.

    Parameters:
    y (array-like): Original labels.
    y_left (array-like): Labels on the left side of the split.
    y_right (array-like): Labels on the right side of the split.

    Returns:
    float: Information gain of the split.
    """
    # 2) TODO: Implement me
    return None

def split(X, y, feature_index, threshold):
    """
    Split the dataset based on a feature and threshold.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Labels.
    feature_index (int): Index of the feature to split on.
    threshold (float): Threshold value to split the feature.

    Returns:
    tuple: (X_left, X_right, y_left, y_right) where
           X_left, X_right are the feature matrices after the split,
           and y_left, y_right are the corresponding labels.
    """

    mask = ~np.isnan(X[:, feature_index])  # Mask to handle missing values
    X_valid = X[mask]  # Features without NaN
    y_valid = y[mask]  # Corresponding labels without NaN

    left_indices = np.where(X_valid[:, feature_index] <= threshold)[0]  # Indices for the left split
    right_indices = np.where(X_valid[:, feature_index] > threshold)[0]  # Indices for the right split
    return X_valid[left_indices], X_valid[right_indices], y_valid[left_indices], y_valid[right_indices]

def best_split(X, y, features):
    """
    Find the best split for the dataset.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Labels.
    features (array-like): List of feature indices to consider for splitting.

    Returns:
    tuple: (best_feature_index, best_threshold) for the best split found.
    """
    best_gain = -1
    best_split = None
    for feature_index in features:
        # If no missing values: thresholds = np.unique(X[:, feature_index])
        thresholds = np.unique(X[:, feature_index][~np.isnan(X[:, feature_index])]) # Unique thresholds without NaNs
        for threshold in thresholds:
            X_left, X_right, y_left, y_right = split(X, y, feature_index, threshold)
            if len(y_left) > 0 and len(y_right) > 0:
                gain = information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, threshold)
    return best_split

def build_tree(X, y, features, depth=0, max_depth=None):
    """
    Build a decision tree.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Labels.
    features (array-like): List of feature indices to consider for splitting.
    depth (int, optional): Current depth of the tree. Default is 0.
    max_depth (int, optional): Maximum depth of the tree. Default is None.

    Returns:
    DecisionNode or Leaf: Root node of the decision tree.
    """
    if len(set(y)) == 1:  # If all labels are the same
        return Leaf(y)
    if max_depth is not None and depth >= max_depth:  # If maximum depth is reached
        return Leaf(y)
    split_result = best_split(X, y, features)
    if split_result is None:  # If no valid split is found
        return Leaf(y)
    feature_index, threshold = split_result
    X_left, X_right, y_left, y_right = split(X, y, feature_index, threshold)
    left = build_tree(
        X_left, y_left, features, depth + 1, max_depth
    )  # Recursively build the left subtree
    right = build_tree(
        X_right, y_right, features, depth + 1, max_depth
    )  # Recursively build the right subtree
    return DecisionNode(feature_index, threshold, left, right)

def predict_sample(node, sample):
    """
    Predict the class of a single sample using the decision tree.

    Parameters:
    node (DecisionNode or Leaf): Root node of the decision tree.
    sample (array-like): Feature values of the sample.

    Returns:
    int: Predicted class label.
    """

    if isinstance(node, Leaf):  # If the node is a leaf
        return node.predictions.most_common(1)[0][0]  # Return the most common label
    
    if np.isnan(sample[node.feature_index]): # If the feature is missing
        left_prediction = predict_sample(node.left, sample) if node.left else None
        right_prediction = predict_sample(node.right, sample) if node.right else None
        
        # Combine predictions from both child nodes, if available
        if left_prediction is not None and right_prediction is not None:
            left_count = node.left.predictions[left_prediction] if isinstance(node.left, Leaf) else 0
            right_count = node.right.predictions[right_prediction] if isinstance(node.right, Leaf) else 0
            return left_prediction if left_count > right_count else right_prediction
        elif left_prediction is not None:
            return left_prediction
        else:
            return right_prediction
    
    # Proceed with normal prediction if no missing value
    if sample[node.feature_index] <= node.threshold:  # If the feature value is less than or equal to the threshold
        return predict_sample(node.left, sample)
    else:
        return predict_sample(node.right, sample)

class RandomForest:
    """
    Random Forest classifier.

    Attributes:
    n_trees (int): Number of trees in the forest.
    max_depth (int): Maximum depth of each tree.
    n_features (int): Number of features to consider for each split.
    trees (list): List of decision trees.
    """
    def __init__(self, n_trees=10, max_depth=None, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        """
        Fit the Random Forest model.

        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Labels.
        """
        # 3) TODO: Implement me

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X (array-like): Feature matrix.

        Returns:
        array: Predicted class labels.
        """
        # 4) TODO: Implement me
        return None

if __name__ == "__main__":
    np.random.seed(0)

    # Load wine dataset
    dataset = load_wine()

    # 6) TODO: Uncomment the following lines to introduce missing values
    # missing_rate=0.1
    # mask = np.random.rand(*dataset.data.shape) < missing_rate
    # dataset.data[mask] = np.nan

    # # Count total NaNs
    # nan_count = np.isnan(dataset.data).sum()
    # print("Number of NaNs in the dataset:", nan_count)

    # # Count NaNs per column
    # nan_count_per_column = np.isnan(dataset.data).sum(axis=0)
    # print("NaNs per column:\n", nan_count_per_column)

    # # Count NaNs per row
    # nan_count_per_row = np.isnan(dataset.data).sum(axis=1)
    # print("NaNs per row:\n", nan_count_per_row)


    # Create train and test split with the ratio 75:25 and print their dimensions
    xtrain, xtest, ytrain, ytest = train_test_split(dataset.data, dataset.target, train_size=0.75, random_state=29) #29

    # Initialize and train Random Forest
    rf = RandomForest(n_trees=100)
    rf.fit(xtrain, ytrain)

    # Predict and evaluate the model
    predictions = rf.predict(xtest)
    accuracy = np.mean(predictions == ytest)
    print(f'Accuracy_own: {accuracy}')

    # 5) TODO: Compare your results to sklearn implementation 

