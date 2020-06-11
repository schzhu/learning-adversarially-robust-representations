import torch
from torch import nn


class BasicEncoder(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer3 = View(-1, 256 * 3 * 3)
        self.layer4 = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.layer5 = nn.Linear(1024, 64)
        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3,
                       self.layer4, self.layer5]

    def forward(self, x: torch.Tensor, return_full_list=False, clip_grad=False,
                prop_limit=None):
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
        for i, layer in enumerate(self.layers):
            if prop_limit is not None and i >= prop_limit:
                break
            x = layer(x)
            if clip_grad:
                x = _clip_grad(x, -clip_grad, clip_grad)
            out.append(x)

        if not return_full_list:
            out = out[-1]

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

basic_encoder = BasicEncoder