# %%
%matplotlib inline
import numpy as np


# %%
x = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

# %%
x = x/np.amax(x, axis=0)
y = y/100

# %%
x

# %%
y.shape

# %%
class Neural_Network(object):
    def __init__(self):
        self.inputLayerSize = 2
        self.outputLayerSize=1
        self.hiddenLayerSize=3

        #weights
        self.w1 = np.random.rand(self.inputLayerSize,self.hiddenLayerSize)
        self.w2 = np.random.rand(self.hiddenLayerSize,self.outputLayerSize)

        print(self.w2)

    def forward(self,x):
        self.z2= np.dot(x,self.w1)
        self.a2= self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w2)
        yHat = self.sigmoid(self.z3)
        return yHat
            
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))  
      

# %%
NN= Neural_Network()
yhats = NN.forward(x)