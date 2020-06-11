'''cortex_DIM losses.

'''

import math

import torch
import torch.nn.functional as F

from functions.gan_losses import get_positive_expectation, get_negative_expectation


def fenchel_dual_loss(l, m, measure=None):
    '''Computes the f-divergence distance between positive and negative joint distributions.

    Note that vectors should be sent as 1x1.

    Divergences supported are Jensen-Shannon `JSD`, `GAN` (equivalent to JSD),
    Squared Hellinger `H2`, Chi-squeared `X2`, `KL`, and reverse KL `RKL`.

    Args:
        l: Local feature map.
        m: Multiple globals feature map.
        measure: f-divergence measure.

    Returns:
        torch.Tensor: Loss.

    '''
    N, units, n_locals = l.size()
    n_multis = m.size(2)

    # First we make the input tensors the right shape.
    l = l.view(N, units, n_locals)
    l = l.permute(0, 2, 1)
    l = l.reshape(-1, units)

    m = m.view(N, units, n_multis)
    m = m.permute(0, 2, 1)
    m = m.reshape(-1, units)

    # Outer product, we want a N x N x n_local x n_multi tensor.
    u = torch.mm(m, l.t())
    u = u.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

    # Since we have a big tensor with both positive and negative samples, we need to mask.
    mask = torch.eye(N).to(l.device)
    n_mask = 1 - mask

    # Compute the positive and negative score. Average the spatial locations.
    # E_pos = get_positive_expectation(u, measure, average=False).min(2)[0].mean(2)
    # E_neg = get_negative_expectation(u, measure, average=False).max(2)[0].mean(2)

    E_pos = get_positive_expectation(u, measure, average=False).mean(2).mean(2)
    E_neg = get_negative_expectation(u, measure, average=False).mean(2).mean(2)

    # Mask positive and negative terms for positive and negative parts of loss
    E_pos = (E_pos * mask).sum() / mask.sum()
    E_neg = (E_neg * n_mask).sum() / n_mask.sum()
    loss = E_neg - E_pos

    return loss


def infonce_loss(l, m):
    '''Computes the noise contrastive estimation-based loss, a.k.a. infoNCE.

    Note that vectors should be sent as 1x1.

    Args:
        l: Local feature map.
        m: Multiple globals feature map.

    Returns:
        torch.Tensor: Loss.

    '''
    N, units, n_locals = l.size() # 16, 2048, 2048
    _, _ , n_multis = m.size() # 16, 2048, 128

    # First we make the input tensors the right shape.
    l_p = l.permute(0, 2, 1) # 16, 2048, 2048
    m_p = m.permute(0, 2, 1) # 16, 128, 2048

    l_n = l_p.reshape(-1, units) # 16*2048, 2048
    m_n = m_p.reshape(-1, units) # 16*128, 2048

    # Inner product for positive samples. Outer product for negative. We need to do it this way
    # for the multiclass loss. For the outer product, we want a N x N x n_local x n_multi tensor.
    u_p = torch.matmul(l_p, m).unsqueeze(2) # 16, 2048, 1, 128
    u_n = torch.mm(m_n, l_n.t())
    u_n = u_n.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)#16, 16, 2048, 128

    # We need to mask the diagonal part of the negative tensor.
    mask = torch.eye(N)[:, :, None, None].to(l.device)
    n_mask = 1 - mask

    # Masking is done by shifting the diagonal before exp.
    u_n = (n_mask * u_n) - 10. * mask  # mask out "self" examples
    u_n = u_n.reshape(N, N * n_locals, n_multis).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

    # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
    pred_lgt = torch.cat([u_p, u_n], dim=2)
    pred_log = F.log_softmax(pred_lgt, dim=2)

    # The positive score is the first element of the log softmax.
    loss = -pred_log[:, :, 0].mean()

    return loss


def donsker_varadhan_loss(l, m, v_out):
    '''

    Note that vectors should be sent as 1x1.

    Args:
        l: Local feature map.
        m: Multiple globals feature map.

    Returns:
        torch.Tensor: Loss.

    '''
    N, units, n_locals = l.size()
    n_multis = m.size(2)

    # First we make the input tensors the right shape.
    l = l.view(N, units, n_locals)
    l = l.permute(1, 0, 2) # hide units as batch
    # l = l.permute(2, 0, 1)
    # l = l.permute(0, 2, 1)
    # l = l.reshape(-1, units)

    m = m.view(N, units, n_multis)
    m = m.permute(1, 2, 0)
    # m = m.permute(2, 1, 0)
    # m = m.permute(0, 2, 1)
    # m = m.reshape(-1, units)

    # Outer product, we want a N x N x n_local x n_multi tensor.
    u = torch.bmm(l, m)
    # u = torch.mm(m, l.t())
    # u = u.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

    # Since we have a big tensor with both positive and negative samples, we need to mask.
    mask = torch.eye(N).to(l.device).unsqueeze(0)
    n_mask = 1 - mask

    # Positive term is just the average of the diagonal.
    # E_pos = (u.mean(2) * mask).sum() / mask.sum()
    # u = u.squeeze()
    # u = u.mean(2).mean(2)
    E_pos = (u * mask).sum((1, 2)) / mask.sum((1, 2))

    # Negative term is the log sum exp of the off-diagonal terms. Mask out the positive.
    u -= 10. * (1 - n_mask)
    u_max = u.max(1, keepdim=True)[0].max(2, keepdim=True)[0]
    # u_max = torch.max(u)
    E_neg = torch.log((n_mask * torch.exp(u - u_max)).sum((1, 2)) + 1e-6) \
            + u_max.squeeze() - math.log(n_mask.sum((1, 2)))
    loss = E_neg - E_pos
    if not v_out:
        loss = loss.sum()

    return loss