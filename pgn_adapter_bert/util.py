"""
Assorted utilities for working with neural networks.
"""

from typing import Tuple, Optional
import torch
from allennlp.common import Registrable, FromParams, Params
from allennlp.nn import util


def batched_linear(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
    """ batched linear forward """
    y = torch.einsum("bth,boh->bto", x, w)
    if b is not None:
        y = y + b.unsqueeze(1)
    return y


def batched_prune(items: torch.Tensor, scores: torch.Tensor,
                  mask: torch.BoolTensor, features: torch.Tensor, topk: int
                  ) -> Tuple[torch.Tensor, ...]:
    """
    Prune based on mention scores.
    """

    # Shape: (batch_size, num_items)
    scores = scores.squeeze(-1)
    # Shape: (batch_size, topk) for all 3 tensors
    top_scores, top_mask, top_indices = util.masked_topk(scores, mask, topk)

    # Shape: (batch_size * topk)
    # torch.index_select only accepts 1D indices, but here we need to select
    # items for each element in the batch. This reformats the indices to take
    # into account their index into the batch. We precompute this here to make
    # the multiple calls to util.batched_index_select below more efficient.
    flat_top_indices = util.flatten_and_batch_shift_indices(top_indices, items.size(1))

    # Compute final predictions for which items to consider as mentions.
    # Shape: (batch_size, topk, *)
    top_items = util.batched_index_select(items, top_indices, flat_top_indices)
    # Shape: (batch_size, topk, feature_size)
    top_features = util.batched_index_select(features, top_indices, flat_top_indices)

    return top_items, top_features, top_mask, top_indices, top_scores, flat_top_indices


def construct_registrable(base: Registrable, *args, **kwargs) -> Registrable:
    """
    Instantiating an object of the registered `kwargs[type]` subclass of `base`
    class by `*args`, `**kwargs`. In this way, you can pass extra parameters
    not only from the json config file, but also python code. It is very helpful
    when the output_dim of some module is varies according to configuration and
    need to be passed to other modules.
    """
    cls = base.by_name(kwargs.pop("type"))
    obj = cls(*args, **kwargs)
    return obj


def construct_from_params(cls: FromParams, **kwargs) -> FromParams:
    """
    Just merge additional kwargs. Such as `input_dim=X, **module`.
    """
    return cls.from_params(Params(kwargs))
