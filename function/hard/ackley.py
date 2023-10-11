import numpy as np
import torch


class Ackley:
    def __init__(self, start_point, a: int = 20, b: int = 0.2, c: int = 2 * np.pi):
        self.x = torch.nn.Parameter(start_point)
        self.a = a
        self.b = b
        self.c = c
        self.dim = len(self.x)

    def forward(self):
        sum_sq_term = -self.a * torch.exp(-self.b * torch.sqrt(torch.mean(self.x ** 2)))
        cos_term = -torch.exp(torch.mean(torch.cos(self.c * self.x)))

        result = self.a + torch.exp(torch.ones(1)) + sum_sq_term + cos_term

        return result
