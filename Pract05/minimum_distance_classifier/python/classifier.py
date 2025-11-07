import numpy as np
from . import mdc_core

class MinimumDistanceClassifier:
    def __init__(self, use_cuda=False):
        self._model = mdc_core.MinimumDistanceClassifier(use_cuda)
    
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        self._model.fit(X, y)
        return self
    
    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self._model.predict(X)
    
    def get_centroids(self):
        return self._model.get_centroids()
    
    @property
    def is_using_cuda(self):
        return self._model.is_using_cuda()
    
    @property
    def is_fitted(self):
        return self._model.is_fitted()
