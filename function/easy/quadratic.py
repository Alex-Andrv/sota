import torch


class Quadratic:
    def __init__(self, start_point: torch.Tensor, diagonal_values: torch.Tensor):
        self.x = torch.nn.Parameter(start_point)
        self.diagonal = torch.diag(diagonal_values)


    def forward(self) -> torch.Tensor:
        # x^T * A * x - вроде torch умеет трнаспонировать
        return self.x.matmul(self.diagonal).matmul(self.x)