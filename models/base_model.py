import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
