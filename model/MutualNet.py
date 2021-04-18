import torch
from torch_geometric.utils import (negative_sampling,
                                   structured_negative_sampling)
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F


class MutualInfoNet(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(MutualInfoNet, self).__init__()
        self.fc_x = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc_y = torch.nn.Linear(1, hidden_channels)
        self.fc =torch.nn.Linear(hidden_channels, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_x.reset_parameters()
        self.fc_y.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, x, y):
        out = F.relu(self.fc_x(x) + self.fc_y(y.unsqueeze(-1)))
        out = self.fc(out)
        return out

