import numpy as np

class ReLU:
    def __call__(self, preActivatedInput):
        if preActivatedInput.ndim == 1:
            preActivatedInput = preActivatedInput.reshape(1, -1)
        return np.where(preActivatedInput<= 0, 0.01, preActivatedInput)
    
    def derivative(self, preActivatedInput):
        if preActivatedInput.ndim == 1:
            preActivatedInput = preActivatedInput.reshape(1, -1)
        return np.where(preActivatedInput<= 0, 0.01, 1)

class Sigmoid:
    def __call__(self, preActivatedInput):
        if preActivatedInput.ndim == 1:
            preActivatedInput = preActivatedInput.reshape(1, -1)
        return 1 / (1 + np.exp(-preActivatedInput))
    
    def derivative(self, preActivatedInput):
        if preActivatedInput.ndim == 1:
            preActivatedInput = preActivatedInput.reshape(1, -1)
        return self(preActivatedInput) * (1 - self(preActivatedInput))

class Softmax:
    def __call__(self, preActivatedInput):
        if preActivatedInput.ndim == 1:
            preActivatedInput = preActivatedInput.reshape(1, -1)
        exped = np.exp(preActivatedInput - np.max(preActivatedInput, axis = 1, keepdims=True))

        return exped / np.sum(exped, axis = 1, keepdims = True)

    def derivative(self, x):
        # Since I use Cross Entropy Loss, MSE Already Returns Desired Derivative
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        return np.ones(x.shape)
    


