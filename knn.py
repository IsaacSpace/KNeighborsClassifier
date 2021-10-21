import numpy as np

def euclidean_distance(x, y):
    assert len(x) == len(y)
    length = len(x)
    dist = 0.0
    for i in range(0, length, 1):
        dist += (x[i] - y[i])**2
    return np.sqrt(dist)

class KNeighbors:
    """K Neighbors 
    This is a standard KNN implementation using euclidean distance
    Parameters
    ----------
    n_neighbors : int 
        numero de vecinos a comparar
    Attributes
    ----------
    train : array
        train dataset
    train_classes : array
        train dataset classes
    """
    def __init__(self, n_neighbors = 100):
        self.neighbors = n_neighbors
        self.train = None
        self.train_classes = None

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples 
            and n_features is the number of features.
        y : array-like of shape (n_samples, )
            Training data targets
        """
        self.train = X
        self.train_classes = y 
    
    def predict(self, X):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Testing data, where n_samples is the number of samples 
            and n_features is the number of features.
        """
        n_vectors, n_features = X.shape
        predictions = []
        for i in range(0, n_vectors, 1):
            distances = []
            for j in range(0, self.train.shape[0], 1):
                distances.append(euclidean_distance(X[i, :], self.train[j, :]))
            sorted_distances = np.argsort(np.array(distances))
            k_neighbors = sorted_distances[0:self.neighbors]
            classes = []
            for k in range(0, len(k_neighbors), 1):
                classes.append(self.train_classes[k_neighbors[k]])
            predictions.append(np.bincount(np.array(classes)).argmax())
        return np.array(predictions)


