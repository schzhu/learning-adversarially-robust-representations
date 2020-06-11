import os
import dill

import torch
from torch import nn

from robustness.tools import helpers
from robustness.attacker import AttackerModel

from models.discriminators import MI1x1ConvNet, PriorDisc, MIFCNet
from models import classifiers
from models.encoder_wrapper import WrappedEncoder
from functions.dim_losses import donsker_varadhan_loss, infonce_loss, fenchel_dual_loss


def make_and_restore_model(args, dataset):
    '''

    '''
    # representation g
    encoder = WrappedEncoder(args, dataset)

    # estimator part 1: X or layer3 to H space
    dim_local = dataset.get_module('Estimator')(args.va_hsize) if not args.share_arch \
                else MI1x1ConvNet(encoder.share_z_size, args.va_hsize)

    # estimator part 2: Z to H space
    dim_global = MI1x1ConvNet(encoder.z_size, args.va_hsize)

    # top classifier
    classifier = classifiers.__dict__[args.classifier_arch]() if \
        args.task == 'train-classifier' else None

    big_encoder = BigEncoder(encoder, dim_local, dim_global, classifier, args)
    atm = AttackerModel(big_encoder, dataset).cuda()
    # atm = torch.nn.DataParallel(atm).cuda()
    # atm = atm.module
    resume_path = args.resume

    checkpoint = None
    if resume_path:
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, pickle_module=dill)

            # Makes us able to load models saved with legacy versions
            state_dict_path = 'model'
            if not ('model' in checkpoint):
                state_dict_path = 'state_dict'
            sd = checkpoint[state_dict_path]

            # lLad from Madry's model. Load encoder, classifier, normalizer only
            sd_est_only_madry = {
                k.replace('module.attacker.model', 'attacker.model.encoder.enc')
                .replace('module.attacker.normalize', 'attacker.normalize'): v
                for k, v in sd.items() if (
                       k.startswith('module.attacker.model') or
                        k.startswith('module.attacker.normalize'))}

            sd_est_only_ours = {
                k: v for k, v in sd.items() if (
                    k.startswith('attacker.model.encoder') or
                    k.startswith('attacker.normalize'))}

            model_dict_est_only = atm.state_dict()
            model_dict_est_only.update(sd_est_only_madry)
            print("=> loaded checkppoint param numbers: {} / {}".format(
                len([_ for _ in sd_est_only_madry.keys()
                 if _ in model_dict_est_only.keys()]),
                len([_ for _ in sd_est_only_madry.keys()])))

            model_dict_est_only.update(sd_est_only_ours)
            print("=> loaded checkppoint param numbers: {} / {}".format(
                len([_ for _ in sd_est_only_ours.keys()
                 if _ in model_dict_est_only.keys()]),
                len([_ for _ in sd_est_only_ours.keys()])))

            # if parallel:
            #     model = torch.nn.DataParallel(model)
            atm.load_state_dict(model_dict_est_only)
            atm = atm.cuda()
            print("=> loaded checkpoint '{}' (epoch {})".format(resume_path,
                checkpoint['epoch']))
        else:
            error_msg = "=> no checkpoint found at '{}'".format(resume_path)
            raise ValueError(error_msg)

    return atm, checkpoint


class BigEncoder(nn.Module):
    '''
    Task: estimate-mi / train-encoder / train-classifier
    '''
    # def __init__(self, scale_prior, lmi, dim_mode, dim_measure, args):
    def __init__(self, encoder, dim_local, dim_global, classifier, args):
        '''
        :param encoder: resnet18 / baseline-encoder /
        :param dim_local: MI1x1ConvNet / OuterEstimator
        :param dim_global:
        :param args:
        '''
        super().__init__()
        self.encoder = encoder
        self.dim_local = dim_local
        self.dim_global = dim_global
        self.classifier = classifier
        self.models = [self.encoder, self.dim_local, self.dim_global, self.classifier]
        self.va_fd_measure = args.va_fd_measure
        self.va_mode = args.va_mode

    def forward(self, input, target, loss_type, detach, enc_in_eval):
        '''
        Compute dim loss or classificaiton loss
        :param input:
        :param loss_type : 'dim' (mi estimation) or 'cla' (classification)
        :param detach:
        :param enc_in_eval:
        :return:
        '''
        prev_training = bool(self.encoder.training)
        if enc_in_eval:
            self.encoder.eval()
        rep_out = self.encoder(input)
        if prev_training:
            self.encoder.train()

        loss_encoder_dim, loss_cla, prec_cla = \
            torch.tensor(0).cuda(), torch.tensor(0).cuda(), torch.tensor(0).cuda()

        if loss_type == 'dim':
            out_local, out_global = self.extract(input, rep_out, self.dim_local,
                                                 self.dim_global)

            loss_encoder_dim = self.cal_dim(out_local, out_global, self.va_fd_measure,
                                            self.va_mode, scale=1.0)

        elif loss_type == 'cla':
            loss_cla, prec_cla = self.cla_eval(rep_out, target, detach=detach)

        else:
            raise NotImplementedError

        return loss_encoder_dim, loss_cla, prec_cla

    def kernel_enc(self, input, no_update_enc=False, prop_limit=2):
        prev_training = bool(self.encoder.training)
        if no_update_enc:
            self.encoder.eval()

        outs = self.encoder(input, return_full_list=True, prop_limit=prop_limit)

        if prev_training:
            self.encoder.train()

        L, G = self.extract_kernel(input, outs)

        loss_prior_disc, prec_prior_disc, loss_encoder_prior = \
            torch.tensor(0).cuda(), torch.tensor(0).cuda(), torch.tensor(0).cuda()

        loss_encoder_dim = self.cal_dim(L, G, self.dim_measure,
                                        self.dim_mode, scale=1.0,
                                        v_out=(self.args.exp2_neuronest_mode==0))
        return loss_prior_disc, prec_prior_disc, loss_encoder_prior, loss_encoder_dim

    @staticmethod
    def custom_loss_func(model, input, target, loss_type=None):
        assert model.training == False
        rep_out = model.encoder(input)

        if loss_type == 'dim':
            out_local, out_global = model.extract(input, rep_out,
                                                  model.dim_local,
                                                  model.dim_global)

            loss_enc_dim = model.cal_dim(out_local, out_global,
                                             model.va_fd_measure,
                                             model.va_mode, scale=1.0)
                                             # v_out=(model.args.exp2_neuronest_mode == 0))
            loss = loss_enc_dim.expand(input.shape[0])

        elif loss_type == 'cla':
            rep_out = model.encoder(input)
            loss_classifier, prec_classifier = model.cla_eval(rep_out,
                                                              target,
                                                              detach=False)  # todo false
            loss = loss_classifier.expand(input.shape[0])

        else:
            raise NotImplementedError

        return loss, None

    def extract(self, input, outs, local_net=None, global_net=None, local_samples=None,
                global_samples=None):
        '''Wrapper function to be put in encoder forward for speed.

        Args:
            outs (list): List of activations
            local_net (nn.Module): Network to encode local activations.
            global_net (nn.Module): Network to encode global activations.

        Returns:
            tuple: local, global outputs

        '''
        L = input
        G = outs
        # All globals are reshaped as 1x1 feature maps.
        global_size = G.size()[1:]
        if len(global_size) == 1:
            G = G[:, :, None, None]
        L = L.detach()
        L = local_net(L)
        G = global_net(G)

        N, local_units = L.size()[:2]
        L = L.view(N, local_units, -1)
        G = G.view(N, local_units, -1)

        # Sample locations for saving memory.
        if global_samples is not None:
            G = sample_locations(G, global_samples)

        if local_samples is not None:
            L = sample_locations(L, local_samples)

        return L, G

    def extract_kernel(self, input, outs, local_net=None, global_net=None,
                local_samples=None, global_samples=None):
        N = input.shape[0]
        NM = N * 8 * 8
        k = 10
        # img, kernel_size, stride, padding
        L = extract_patch(input, k, 4, 3) # (8192, 3, 10, 10)
        G = extract_patch(outs[1], 1, 1, 0) # (8192, 128, 1, 1)

        # All globals are reshaped as 1x1 feature maps.
        global_size = G.size()[1:]
        if len(global_size) == 1:
            G = G[:, :, None, None]

        L = L.detach()
        L = self.dim_local(L) # 2048, 2048, 4, 4 -> 16, 2048, 128*4*4
        G = self.dim_global(G)

        _, units = L.size()[:2]

        L = L.view(NM, units, 1)
        G = G.view(NM, units, 1)

        # Sample locations for saving memory.
        if global_samples is not None:
            G = sample_locations(G, global_samples)

        if local_samples is not None:
            L = sample_locations(L, local_samples)

        return L, G

    def cla_eval(self, outs, target, detach=True,
                            criterion=nn.CrossEntropyLoss()):
        '''

        Args:
            outs
            target
            criterion

        Returns:
            tuple: loss, prec

        '''
        if self.classifier is not None:
            output = self.classifier(outs)
        else:
            output = outs

        loss = criterion(output, target)
        prec1 = helpers.accuracy(output, target)[0]

        return loss, prec1


    def cal_dim(self, L, G, measure='JSD', mode='fd', scale=1.0, act_penalty=0.,
                v_out=False):
        '''

        Args:
            measure: Type of f-divergence. For use with mode `fd`.
            mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
            scale: Hyperparameter for local DIM. Called `beta` in the paper.
            act_penalty: L2 penalty on the global activations. Can improve stability.

        '''

        if mode == 'fd':
            loss = fenchel_dual_loss(L, G, measure=measure)
        elif mode == 'nce':
            loss = infonce_loss(L, G)
        elif mode == 'dv':
            loss = donsker_varadhan_loss(L, G, v_out)
        else:
            raise NotImplementedError(mode)

        # if scale > 0:
        #     self.add_losses(encoder=scale * loss + act_loss)
        loss_encoder = scale * loss

        return loss_encoder

    def _update_model_list(self):
        self.models = [self.encoder, self.dim_local, self.dim_global,
                       self.prior_disc, self.classifier]

def sample_locations(enc, n_samples):
    '''Randomly samples locations from localized features.

    Used for saving memory.

    Args:
        enc: Features.
        n_samples: Number of samples to draw.

    Returns:
        torch.Tensor

    '''
    n_locs = enc.size(2)
    batch_size = enc.size(0)
    weights = torch.tensor([1. / n_locs] * n_locs, dtype=torch.float)
    idx = torch.multinomial(weights, n_samples * batch_size, replacement=True) \
        .view(batch_size, n_samples)
    enc = enc.transpose(1, 2)
    adx = torch.arange(0, batch_size).long()
    enc = enc[adx[:, None], idx].transpose(1, 2)

    return enc


def extract_patch(img, kernel_size, stride, padding):
    '''
    https://discuss.pytorch.org/t/how-to-extract-smaller-image-patches-3d/16837/2
    https://gist.github.com/dem123456789/23f18fd78ac8da9615c347905e64fc78
    :param img: (N, C, H, W), e.g. (N, 3, 32, 32) (128, 256, 8, 8)

    :return: (N * down_scale**2, C, H / down_scale, W / down_scale)
    '''
    N, C, H, W = img.shape
    row_patch_num = (H + 2 * padding - kernel_size) // stride + 1
    pad = nn.ConstantPad2d(padding, 0)
    out = pad(img)
    # unfold(dimension, size, step)
    out = out.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    out = out.permute(0, 2, 3, 1, 4, 5)
    out = out.contiguous().view(N * row_patch_num**2, C, kernel_size, kernel_size)
    return out
