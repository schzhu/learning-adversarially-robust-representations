import torch
from torch import nn


class WrappedEncoder(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()
        self.enc = dataset.get_model(args.arch, False)
        if args.arch == 'encoder':
            self.forward = self.forward_baseline
        else:
            self.forward = self.forward_canonical

        self.share_z_size = 256
        self.z_size = self._get_z_size(args)

    def forward_baseline(self, input):
        out = self.enc(input, return_full_list=False) # , prop_limit=prop_limit)
        return out

    def forward_canonical(self, input):
        out = self.enc(input) #, with_latent=True)
        return out

    def _get_z_size(self, args):
        if (args.task == 'train-encoder') or (args.task == 'train-classifier'):
            return 64
        else:
            return 10