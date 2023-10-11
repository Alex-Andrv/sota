import torch

from function.function import Function


class SixHumpCamel(Function):

    def __init__(self, start_point: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = torch.nn.Parameter(start_point)

    def calculate(self, x: torch.Tensor) -> torch.Tensor:
        term1 = 4*x[0] ** 2-2.1*x[0]**4+1/3*x[0]**6
        term2 = x[0] *x[1]
        term3 = -4*(x[1]**2-x[1]**4)
        return term1 + term2 + term3