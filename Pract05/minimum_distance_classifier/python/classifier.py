import numpy as np
from . import mdc_core

class MinimumDistanceClassifier:
    def __init__(self, use_cuda=True):
        #  instanciate C++ object
        self._model = mdc_core.MinimumDistanceClassifier(use_cuda)

    def fit(self, X, y):
        # numpy to lists if necesarry
        if isinstance(X, np.ndarray):
            X_list = X.astype(np.float32).tolist()
        else:
            X_list = X
            
        if isinstance(y, np.ndarray):
            y_list = y.astype(np.int32).tolist()
        else:
            y_list = y
        
        # call C++ method
        self._model.fit(X_list, y_list)
        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X_list = X.astype(np.float32).tolist()
        else:
            X_list = X
            
        # call to the C++ method, returns python list
        preds_list = self._model.predict(X_list)
        
        # result to numpy array
        return np.array(preds_list, dtype=np.int32)

    def get_centroids(self):
        
        # call to C++ method; returns a list of lists
        centroids_list = self._model.get_centroids()
        
        # to numpy array
        return np.array(centroids_list, dtype=np.float32)
