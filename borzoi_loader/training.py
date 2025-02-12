from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from borzoi_pytorch import Borzoi


# Following function is taken from https://github.com/gagneurlab/scooby/blob/main/scooby/utils/utils.py
# (MIT license, Copyright (c) 2024 Gagneur lab: https://github.com/gagneurlab/scooby/blob/main/LICENSE)
def poisson_multinomial_torch(
    y_pred,
    y_true,
    total_weight: float = 0.2,
    epsilon: float = 1e-6,
    rescale: bool = False,
):
    """
    Calculates the Poisson-Multinomial loss.

    This loss function combines a Poisson loss term for the total count and a multinomial loss term for the 
    distribution across sequence positions.

    Args:
        y_pred (torch.Tensor): Predicted values (batch_size, seq_len).
        y_true (torch.Tensor): True values (batch_size, seq_len).
        total_weight (float, optional): Weight of the Poisson total term. Defaults to 0.2.
        epsilon (float, optional): Small value added to avoid log(0). Defaults to 1e-6.
        rescale (bool, optional): Whether to rescale the loss. Defaults to False.

    Returns:
        torch.Tensor: The mean Poisson-Multinomial loss.
    """
    seq_len = y_true.shape[1]

    # add epsilon to protect against tiny values
    y_true += epsilon
    y_pred += epsilon

    # sum across lengths
    s_true = y_true.sum(dim=1, keepdim=True)
    s_pred = y_pred.sum(dim=1, keepdim=True)

    # normalize to sum to one
    p_pred = y_pred / s_pred

    # total count poisson loss
    poisson_term = F.poisson_nll_loss(s_pred, s_true, log_input=False, eps=0, reduction="mean")  # B x T
    # print (poisson_term,poisson_term.shape)
    poisson_term /= seq_len
    # print (poisson_term)

    # multinomial loss
    pl_pred = torch.log(p_pred)  # B x L x T
    multinomial_dot = -torch.multiply(y_true, pl_pred)  # B x L x T
    multinomial_term = multinomial_dot.sum(dim=1)  # B x T
    multinomial_term /= seq_len
    # print (multinomial_term.mean(), poisson_term.mean())

    # normalize to scale of 1:1 term ratio
    loss_raw = multinomial_term + total_weight * poisson_term
    # print (loss_raw.shape)
    if rescale:
        loss_rescale = loss_raw * 2 / (1 + total_weight)
    else:
        loss_rescale = loss_raw

    return loss_rescale.mean()



# Note about code origin
# Most of the next code is adapted from both https://github.com/johahi/borzoi-pytorch/tree/main
# which was released under Apache license 2.0 - https://github.com/johahi/borzoi-pytorch/tree/main?tab=Apache-2.0-1-ov-file#readme
# and https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/finetune.py
# which was released under MIT license Copyright (c) 2021 Phil Wang https://github.com/lucidrains/enformer-pytorch/tree/main?tab=MIT-1-ov-file#readme

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))


def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_batchnorms_(model):
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]

    for bn in bns:
        bn.eval()
        bn.track_running_stats = False
        set_module_requires_grad_(bn, False)

def freeze_all_but_layernorms_(model):
    for m in model.modules():
        set_module_requires_grad_(m, isinstance(m, nn.LayerNorm))



class HeadAdapterWrapper(nn.Module):
    def __init__(
        self,
        *,
        borzoi,
        num_tracks = 1, # Default to predicting one track
        output_activation: Optional[nn.Module] = nn.Softplus(),
        auto_set_target_length = True
    ):
        super().__init__()
        assert isinstance(borzoi, Borzoi)
        self.borzoi = borzoi
        # freeze the batch norms
        freeze_batchnorms_(self.borzoi)

        self.auto_set_target_length = auto_set_target_length

        borzoi_hidden_dim = 1920

        # Prediction head to be trained
        self.to_tracks = Sequential(
            nn.Conv1d(in_channels = borzoi_hidden_dim,
                      out_channels = num_tracks,
                      kernel_size = 1),
            output_activation
        )

        # Freeze all model parameters except `to_tracks`
        for param in self.parameters():
            param.requires_grad = False  # Disable gradients for everything

        # Enable gradients only for `to_tracks`
        for param in self.to_tracks.parameters():
            param.requires_grad = True  # Enable gradients for this part

    def forward(
        self,
        seq,
        *,
        target = None
    ):
        borzoi_kwargs = dict()

        if exists(target) and self.auto_set_target_length:
            borzoi_kwargs = dict(target_length = target.shape[-2])

        embeddings = self.borzoi.get_embs_after_crop(seq)
        embeddings = self.borzoi.final_joined_convs(embeddings)

        # embeddings = get_borzoi_embeddings(self.borzoi, seq, freeze = freeze_enformer, train_layernorms_only = finetune_enformer_ln_only, train_last_n_layers_only = finetune_last_n_layers_only, enformer_kwargs = enformer_kwargs)

        preds = self.to_tracks(embeddings).transpose(2,1)

        if not exists(target):
            return preds

        return poisson_multinomial_torch(preds, target)

