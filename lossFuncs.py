import numpy as np

class MSE:
    def __call__(self, modelOutput, expected):
        return np.sum((2 * (modelOutput - expected) / modelOutput.shape[0]), axis = 0)
    

