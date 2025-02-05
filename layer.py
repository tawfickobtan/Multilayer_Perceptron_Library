import numpy as np
from activationFuncs import *

class layer:
    def __init__(self, inputs, outputs, wantBias = False, activationFunc = None, alpha = 0.01):
        self.inputs = inputs
        self.outputs = outputs
        self.bias = wantBias
        self.activation = activationFunc
        self.weights = np.random.rand(inputs, outputs)
        self.alpha = alpha

        if wantBias:
            self.weights = np.vstack((self.weights, np.ones((1, outputs))))
        print("Initial weights:")
        print(self.weights)

        self.lastInput = None
        self.lastOutput = None
        self.updateWeights = np.zeros(self.weights.shape)
        print(self.update)
        
    
    def run(self, input):
        batchSize, features = 1, 1
        if len(input.shape) == 1:
            print("Single input")
            features = input.shape[0]
        else:
            print("Batch")
            batchSize, features = input.shape
        if features != self.inputs:
            print("Feature count incorrect")
            print("correct:", self.inputs)
            print("Your input:", features)
            return None
        
        self.lastInput = np.copy(input)

        if self.bias:
            input = np.hstack((input, np.ones((batchSize, 1))))

        res = input @ self.weights
        
        if self.activation:
            res = self.activation(res)

        self.lastOutput = np.copy(res)

        return res
        
    def back(self, derivativeSoFar):
        # derivativeSoFar is the vector of the derivative of the loss function with respect to the
        # output of each neuron in this layer
        if self.activation:
            derivativeSoFar *= self.activation.derivative(self.lastOutput)
        self.updateWeights += self.lastInput.T @ derivativeSoFar

        return derivativeSoFar @ self.weights.T

    def update(self):
        self.weights -= self.alpha * self.update
        self.updateWeights = np.zeros(self.weights.shape)
        


