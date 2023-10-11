import torch


class Beale:

    def __init__(self, start_point: torch.Tensor, a: float = 1.5, b: float = 2.25, c: float = 2.625):
        self.x = torch.nn.Parameter(start_point)
        self.a = a
        self.b = b
        self.c = c

    def forward(self) -> torch.Tensor:
        term1 = (self.a - self.x[0] + self.x[0] * self.x[1]) ** 2
        term2 = (self.b - self.x[0] + self.x[0] * self.x[1] ** 2) ** 2
        term3 = (self.c - self.x[0] + self.x[0] * self.x[1] ** 3) ** 2
        return term1 + term2 + term3