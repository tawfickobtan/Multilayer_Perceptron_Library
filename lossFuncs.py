import numpy as np

class ReLU:
    def __call__(self, preActivatedInput):
        return np.where(preActivatedInput<= 0, 0, preActivatedInput)
    
temp = ReLU()
print(temp(np.array([[1, 2, 3], 
           [-1, 2, -5], 
           [5, 3, -9]])))