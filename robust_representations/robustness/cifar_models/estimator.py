import numpy as np
import torch
import torch.nn as nn


class Estimator(nn.Module):
    def __init__(self, n_output, cnn_input=128):
        n_input = cnn_input
        n_units = n_output
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        # )
        # self.layer3 = View(-1, 256 * 4 * 4)
        # self.layer4 = nn.Sequential(
        #     nn.Linear(4096, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        # )
        # self.layer5 = nn.Linear(1024, 64)
        self.layers = [self.layer0, self.layer1]
        # self.layers = [self.layer0, self.layer1, self.layer2, self.layer3,
        #                self.layer4, self.layer5]
        self.block_nonlinear = nn.Sequential(
            nn.Conv2d(n_input, n_units, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_units),
            nn.ReLU(),
            nn.Conv2d(n_units, n_units, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.block_ln = nn.Sequential(
            Permute(0, 2, 3, 1),
            nn.LayerNorm(n_units),
            Permute(0, 3, 1, 2)
        )

        self.linear_shortcut = nn.Conv2d(n_input, n_units, kernel_size=1,
                                         stride=1, padding=0, bias=False)

        # initialize shortcut to be like identity (if possible)
        if n_units >= n_input:
            eye_mask = np.zeros((n_units, n_input, 1, 1), dtype=np.uint8)
            for i in range(n_input):
                eye_mask[i, i, 0, 0] = 1
            self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
            self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

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

        # if not return_full_list:
        out = out[-1]
        out = self.block_ln(self.block_nonlinear(out) + self.linear_shortcut(out))

        return out


class MINIConvNet(nn.Module):
    """Simple custorm 1x1 convnet.

    """
    def __init__(self, img_size, n_input, n_units_0, n_units_1, n_output):
        """

        Args:
            n_input: Number of input units.
            n_units: Number of output units.
        """

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_input, n_units_0, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_units_0),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_units_0, n_units_1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_units_1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.fc = nn.Sequential(
            nn.Linear((img_size // 2)**2 * n_units_1, n_output)
        )

    def forward(self, x):
        """

            Args:
                x: Input tensor.

            Returns:
                torch.Tensor: network output.

        """

        h = self.conv1(x)
        h = self.conv2(h)
        h = h.view(h.shape[0], -1)
        h = self.fc(h)
        return h


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


class Permute(torch.nn.Module):
    """Module for permuting axes.

    """
    def __init__(self, *perm):
        """

        Args:
            *perm: Permute axes.
        """
        super().__init__()
        self.perm = perm

    def forward(self, input):
        """Permutes axes of tensor.

        Args:
            input: Input tensor.

        Returns:
            torch.Tensor: permuted tensor.

        """
        return input.permute(*self.perm)