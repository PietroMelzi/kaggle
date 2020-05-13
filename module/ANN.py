from torch import nn
import torch.nn.functional as F

I = 40
H = 80
O = 2


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(I, H)
        self.output = nn.Linear(H, O)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.output(x)

        return x


