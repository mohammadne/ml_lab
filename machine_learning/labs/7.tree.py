import numpy as np

def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1- p)*np.log2(1 - p)

def split_indices(X, index_feature):
    """Given a dataset and a index feature, return two lists for the two split nodes, the left node has the animals that have
    that feature = 1 and the right node those that have the feature = 0
    index feature = 0 => ear shape
    index feature = 1 => face shape
    index feature = 2 => whiskers
    """
    left_indices = []
    right_indices = []
    for i,x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices

def weighted_entropy(X,y,left_indices,right_indices):
    """
    This function takes the splitted dataset, the indices we chose to split and returns the weighted entropy.
    """
    w_left = len(left_indices)/len(X)
    w_right = len(right_indices)/len(X)
    p_left = sum(y[left_indices])/len(left_indices)
    p_right = sum(y[right_indices])/len(right_indices)

    weighted_entropy = w_left * entropy(p_left) + w_right * entropy(p_right)
    return weighted_entropy

def information_gain(X, y, left_indices, right_indices):
    """
    Here, X has the elements in the node and y is theirs respectives classes
    """
    p_node = sum(y)/len(y)
    h_node = entropy(p_node)
    w_entropy = weighted_entropy(X,y,left_indices,right_indices)
    return h_node - w_entropy


class Node:
    def __init__(self, feature=None, prediction=None):
        self.feature = feature
        self.left = None
        self.right = None
        self.prediction = prediction

    def build_tree(self, X, y, features):
        # Stop if pure node
        if len(set(y)) == 1:
            self.prediction = y[0]
            return

        # Stop if no features left
        if not features:
            self.prediction = int(np.round(np.mean(y)))
            return

        # Find best feature
        best_gain = -1
        best_feature = None
        best_split = None

        for f in features:
            left_idx, right_idx = split_indices(X, f)
            gain = information_gain(X, y, left_idx, right_idx)

            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_split = (left_idx, right_idx)

        # If no useful split
        if best_gain <= 0:
            self.prediction = int(np.round(np.mean(y)))
            return

        self.feature = best_feature

        left_idx, right_idx = best_split

        # Recursive build
        self.left = Node()
        self.left.build_tree(
            X[left_idx],
            y[left_idx],
            [f for f in features if f != best_feature],
        )

        self.right = Node()
        self.right.build_tree(
            X[right_idx],
            y[right_idx],
            [f for f in features if f != best_feature],
        )

    def print_tree(self, feature_names, indent=""):
        if self.prediction is not None:
            print(f"{indent}Predict → {self.prediction}")
            return

        feature_name = feature_names[self.feature]

        print(f"{indent}if {feature_name} == 1:")
        self.left.print_tree(feature_names, indent + "  ")

        print(f"{indent}else ({feature_name} == 0):")
        self.right.print_tree(feature_names, indent + "  ")

if __name__ == "__main__":
    feature_names = ["Ear Shape", "Face Shape", "Whiskers"]
    # Ear Shape (1 if pointy, 0 otherwise)
    # Face Shape (1 if round, 0 otherwise)
    # Whiskers (1 if present, 0 otherwise)
    X_train = np.array([
    [1, 1, 1],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
    [1, 1, 0],
    [0, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 0]])

    y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
   
    root = Node()
    root.build_tree(X_train, y_train, features=[0, 1, 2])
    root.print_tree(feature_names)
