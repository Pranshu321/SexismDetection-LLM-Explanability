import torch
import torch.nn as nn
from norse.torch import LIFCell, SequentialState  # Import necessary modules

# Define the same model architecture as the one used for saving


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = SequentialState(
            nn.Linear(512, 256),
            LIFCell(),
            nn.Linear(256, 64),
            LIFCell(),
            nn.Linear(64, 1),
            LIFCell(),
        )

    def forward(self, x):
        output, state = self.model(x)
        return output


# Initialize model
model = MyModel()

# Load weights
model.load_state_dict(torch.load('C:/Users/prans/Desktop/M.Tech Sem/AIDA Sem2/NN/Project/SexismLLM/App/spike_model.pth'))
model.eval()  # Set to evaluation mode
