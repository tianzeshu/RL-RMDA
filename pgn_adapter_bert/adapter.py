from typing import Union
import torch
import torch.nn as nn
from torch.nn.functional import linear
from transformers.models.bert.modeling_bert import BertOutput, BertSelfOutput
from allennlp.nn import Activation
from .util import batched_linear


class Adapter(nn.Module):

    def __init__(self, in_features: int, adapter_size: int = 64, bias: bool = True,
                 activation: str = 'gelu', external_param: bool = False,
                 train_layer_norm: bool = True):
        super().__init__()
        self.in_features = in_features
        self.adapter_size = adapter_size
        self.bias = bias
        self.train_layer_norm = train_layer_norm
        self.act_fn = Activation.by_name(activation)()  # GELU is the best one.

        if external_param:
            self.weight_down, self.weight_up = None, None
        else:
            self.weight_down = nn.Parameter(torch.Tensor(adapter_size, in_features))
            self.weight_up = nn.Parameter(torch.Tensor(in_features, adapter_size))
            self.reset_parameters()

        if external_param or not bias:
            self.bias_down, self.bias_up = None, None
        else:
            self.bias_down = nn.Parameter(torch.zeros(adapter_size))
            self.bias_up = nn.Parameter(torch.zeros(in_features))

    def reset_parameters(self):
        nn.init.normal_(self.weight_down, std=1e-3)
        nn.init.normal_(self.weight_up, std=1e-3)

    def forward(self, hidden_states: torch.Tensor):
        linear_func = batched_linear if self.weight_down.dim() == 3 else linear
        x = linear_func(hidden_states, self.weight_down, self.bias_down)
        x = self.act_fn(x)
        x = linear_func(x, self.weight_up, self.bias_up)
        x = x + hidden_states
        return x


class AdapterBertOutput(nn.Module):
    def __init__(self, base: Union[BertOutput, BertSelfOutput], adapter: Adapter):
        super().__init__()
        self.base = base
        self.adapter_forward = adapter.forward
        for param in base.LayerNorm.parameters():
            param.requires_grad = adapter.train_layer_norm

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.base.dense(hidden_states)
        hidden_states = self.base.dropout(hidden_states)
        hidden_states = self.adapter_forward(hidden_states)
        hidden_states = self.base.LayerNorm(hidden_states + input_tensor)
        return hidden_states
