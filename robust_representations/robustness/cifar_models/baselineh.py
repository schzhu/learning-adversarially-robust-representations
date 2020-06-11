import torch
from torch import nn

from robustness.cifar_models.encoder import BasicEncoder
from models.classifiers import Classifier, LinearClassifier
from functools import partial


class BaselineH(nn.Module):
    def __init__(self, num_classes=None, cla_type=None):
        super().__init__()
        self.base_enc = BasicEncoder()
        self.base_cla = Classifier() if cla_type == 'mlp' else LinearClassifier()

    def forward(self, x: torch.Tensor):
        '''Forward pass

        Args:
            x: Input.
            return_full_list: Optional, returns all layer outputs.

        Returns:
            torch.Tensor or list of torch.Tensor.

        '''
        out = self.base_enc(x)
        out = self.base_cla(out)

        return out


baseline_mlp = partial(BaselineH, cla_type='mlp')
baseline_linear = partial(BaselineH, cla_type='linear')