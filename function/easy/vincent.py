import torch


class Vincent:
    def __init__(self, start_point: torch.Tensor):
        self.x = torch.nn.Parameter(start_point)

    def forward(self) -> torch.Tensor:
        term1 = (4 - 2.1 * self.x[0] ** 2 + (self.x[0] ** 4) / 3) * self.x[0] ** 2
        term2 = self.x[0] * self.x[1]
        term3 = (-4 + 4 * self.x[1] ** 2) * self.x[1] ** 2
        return term1 + term2 + term3