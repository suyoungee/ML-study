import torch
import torch.nn as nn # neural network
import torch.nn.functional as F
import torch.optim as optim # optimizer

# create neural network model
# input -> hidden : (3,5)
# hidden -> output : (5,2)
class first_model(nn.Module):
    def __init__(self) :
        super(first_model, self).__init__()
        # set weight matrix 1
        self.lin1 = nn.Linear(3,5)
        # set weight matrix 2
        self.lin2 = nn.Linear(5,2)
    
    # forward propagation
    def forward(self, x) : # x: input data
        x = self.lin1(x)
        x = F.relu(x) # ReLU(Rectified Linear Unit)
        x = self.lin2(x)
        x = F.sigmoid(x)
        return x

# define neural network model and optimize
model = first_model()
opt = optim.SGD(model.parameters(), lr=0.01) # Stochastic Gradient Descent

print(model)

import numpy as np

# Backward propagation
criterion = nn.MSELoss() # MSELoss : (y_-y)**2

x = torch.Tensor(np.random.normal(size=(3))) # input data (3D vector)
y = torch.Tensor(np.random.normal(size=(2))) # output data (2D vector)

opt.zero_grad() # initialize all gradients to 0
y_infer = model(x) # Forward propagation
loss = criterion(y_infer, y)
loss.backward()
opt.step()