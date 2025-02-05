import numpy as np

class ReLU:
    def __call__(self, preActivatedInput):
        return np.where(preActivatedInput<= 0, 0, preActivatedInput)
    
    def derivative(self, preActivatedInput):
        return np.where(preActivatedInput<= 0, 0, 1)

class Sigmoid:
    def __call__(self, preActivatedInput):
        return 1 / (1 + np.exp(-preActivatedInput))
    
    def derivative(self, preActivatedInput):
        return self(preActivatedInput) * (1 - self(preActivatedInput))
