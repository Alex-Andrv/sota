import torch

from function.function import Function


class NewFunction02(Function):

    def __init__(self, start_point: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = torch.nn.Parameter(start_point)

    def calculate(self, x: torch.Tensor) -> torch.Tensor:

        return torch.abs(torch.sin(torch.sqrt(torch.abs(x[0]**2+x[1]))))**(0.5)+(x[0]+x[1])/100