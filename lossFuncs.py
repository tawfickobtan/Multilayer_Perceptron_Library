import numpy as np

class MSE:
    def __call__(self, modelOutput, expected):
        return 2 * (modelOutput - expected)
    

