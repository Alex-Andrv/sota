import torch

from function.function import Function


class Deb02(Function):

    def __init__(self, start_point: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = torch.nn.Parameter(start_point)

    def sumElem(self,i:int,x:torch.Tensor):
        return torch.sin(5*torch.pi*(abs(x[i])**(3/4)-0.05))
    
    def calculate(self, x: torch.Tensor) -> torch.Tensor:
        sum = 0
        for i in range(2):
            sum+=self.sumElem(i,x)
        return sum
    