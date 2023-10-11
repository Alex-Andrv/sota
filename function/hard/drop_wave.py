import numpy as np
import torch


class DropWave:
    def __init__(self, start_point):
        self.x = torch.nn.Parameter(start_point)

    def forward(self):
        norm_squared = torch.sum(self.x ** 2)
        numerator = 1 + torch.cos(12 * torch.sqrt(norm_squared))
        denominator = 0.5 * (norm_squared) + 2
        result = -numerator / denominator
        return result
