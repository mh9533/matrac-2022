import numpy as np

# Interface
class Model:
    def fit(self, X, y): return self 
    def predict(self, X): raise NotImplementedError

class Identity(Model):
    """Predict by copying input column to output."""

    def __init__(self, column_idx=0):
        self.column_idx = column_idx

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[:, self.column_idx]

class MeanOffsetModel(Model):
    def __init__(self):
        self.offset = 0

    def fit(self, X, y):
        self.offset = np.mean(y - X[:, 0])
        return self
    
    def predict(self, X):
        return X[:, 0] + self.offset

class ColumnMean(Model):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.mean(X, axis=1)
