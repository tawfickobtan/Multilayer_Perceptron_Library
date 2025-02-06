import numpy as np
from layer import *
from lossFuncs import *

class model:
    def __init__(self, *layers, loss = MSE()):
        self.layers = []
        self.loss = loss
        for layer in layers:
            self.layers.append(layer)
    
    def printLayers(self):
        for layer in self.layers:
            print(layer)
    
    def run(self, input):
        output = np.copy(input)
        for layer in self.layers:
            output = layer.run(output)
        return output
        
    def back(self, input, expectedOutput):
        derivativeSoFar = self.loss(self.run(input), expectedOutput)

        for layer in self.layers[::-1]:
            derivativeSoFar = layer.back(derivativeSoFar)
    



