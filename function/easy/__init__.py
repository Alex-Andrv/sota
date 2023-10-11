import numpy as np
import torch
class Bohachevsky:
    def __init__(self, start_point, a: int = 20, b: int = 0.2, c: int = 2 * np.pi):
        self.x = torch.nn.Parameter(start_point)
        self.a = a
        self.b = b
        self.c = c
        self.dim = len(self.x)

    def forward(self):
        term1 = x1 ** 2
        term2 = 2 * x2 ** 2
        term3 = -0.3 * tf.cos(3 * tf.pi * x1)
        term4 = -0.4 * tf.cos(4 * tf.pi * x2)
        return term1 + term2 + term3 + term4 + 0.7
