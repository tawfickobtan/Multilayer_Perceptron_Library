import numpy as np

class layer:
    def __init__(self, inputs, outputs, wantBias = False, activationFunc = None):
        self.inputs = inputs
        self.outputs = outputs
        self.bias = wantBias
        self.activation = activationFunc
        self.weights = np.random.rand(inputs, outputs)

        if wantBias:
            self.weights = np.vstack((self.weights, np.ones((1, outputs))))
        print("Initial weights:")
        print(self.weights)
    
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

        if self.bias:
            input = np.hstack((input, np.ones((batchSize, 1))))

        res = input @ self.weights

        return res

mylayer = layer(3, 1, wantBias = True)
print(mylayer.run(np.array(([3, 2, 5],[1, 2, 3]))))


        
        
