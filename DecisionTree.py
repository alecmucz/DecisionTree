import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature  # Feature index
        self.threshold = threshold  # Threshold for splitting
        self.left = left  # Left subtree
        self.right = right  # Right subtree
        self.value = value  # Leaf node value

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)
        
        if len(unique_labels) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return Node(value=Counter(y).most_common(1)[0][0])
        
        best_feature, best_threshold = self._best_split(X, y, num_features)
        
        if best_feature is None:
            return Node(value=Counter(y).most_common(1)[0][0])
        
        left_idx = X[:, best_feature] < best_threshold
        right_idx = ~left_idx
        left_child = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    
    def _best_split(self, X, y, num_features):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_y, right_y = y[X_column < threshold], y[X_column >= threshold]
        n , n_left, n_right = len(y), len(left_y), len(right_y)
        if n_left == 0 or n_right == 0:
            return 0
        child_entropy = (n_left/n) * self._entropy(left_y) + (n_right/n) * self._entropy(right_y)
        return parent_entropy - child_entropy

    def _entropy(self, y):
        counts = np.bincount(y)
        probabilites = counts/np.sum(counts)
        return -np.sum([p*np.log2(p) for p in probabilites if p>0])

    def _predict(self, X):
        return np.array([self._traverse_tree(x,self.root) for x in X])

    def _traverse(self, x, node:Node):
        if node.is_leaf():
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse(x, node.left)
        if x[node.feature] >= node.threshold:
            return self._traverse(x, node.right)

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tree = DecisionTree(max_depth=3)
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)

    print(predictions)
    print(y_test)
    
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy:{accuracy:.2f}")