import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        self.layers = nn.Sequential(
            nn.Linear(in_features=774, out_features=2048),
            nn.ELU(),
            #nn.BatchNorm1d(2048),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ELU(),
            #nn.BatchNorm1d(2048),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ELU(),
            #nn.BatchNorm1d(2048),
            nn.Linear(in_features=2048, out_features=1)
        )

    def forward(self, x):
        return self.layers(x)
