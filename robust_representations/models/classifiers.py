import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(64, 200),
            nn.Dropout(0.1),
            nn.BatchNorm1d(200),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Linear(200, 10),
        )

        self.layers = [self.layer0, self.layer1]

    def forward(self, x: torch.Tensor, return_full_list=False, clip_grad=False):
        '''Forward pass

        Args:
            x: Input.
            return_full_list: Optional, returns all layer outputs.

        Returns:
            torch.Tensor or list of torch.Tensor.

        '''

        def _clip_grad(v, min, max):
            v_tmp = v.expand_as(v)
            v_tmp.register_hook(lambda g: g.clamp(min, max))
            return v_tmp

        out = []
        for layer in self.layers:
            x = layer(x)
            if clip_grad:
                x = _clip_grad(x, -clip_grad, clip_grad)
            out.append(x)

        if not return_full_list:
            out = out[-1]

        return out


class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor):
        out = self.layer0(x)
        return out


class View(torch.nn.Module):
    """Basic reshape module.

    """
    def __init__(self, *shape):
        """

        Args:
            *shape: Input shape.
        """
        super().__init__()
        self.shape = shape

    def forward(self, input):
        """Reshapes tensor.

        Args:
            input: Input tensor.

        Returns:
            torch.Tensor: Flattened tensor.

        """
        return input.view(*self.shape)


mlp = Classifier
linear = LinearClassifier