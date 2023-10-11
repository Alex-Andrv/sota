import torch


class Bohachevsky:

    def __init__(self, start_point: torch.Tensor):
        self.x = torch.nn.Parameter(start_point)

    def forward(self) -> torch.Tensor:
        term1 = self.x[0] ** 2
        term2 = 2 * self.x[1] ** 2
        term3 = -0.3 * torch.cos(3 * torch.tensor(2 * torch.pi) * self.x[0])
        term4 = -0.4 * torch.cos(4 * torch.tensor(2 * torch.pi) * self.x[1])
        return term1 + term2 + term3 + term4 + 0.7