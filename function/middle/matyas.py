import torch


class Matyas:
    def __init__(self, start_point: torch.Tensor, a: float = 0.26, b: float = 0.48):
        self.x = torch.nn.Parameter(start_point)
        self.a = a
        self.b = b

    def forward(self) -> torch.Tensor:
        term1 = self.a * (self.x[0] ** 2 + self.x[1] ** 2)
        term2 = self.b * self.x[0] * self.x[1]
        return term1 - term2