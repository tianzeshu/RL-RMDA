from typing import Optional, Dict, Any, Union, List
import torch.nn as nn
from transformers import BertModel, ElectraModel, RobertaModel
from allennlp.common.checks import ConfigurationError
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from .adapter import Adapter, AdapterBertOutput


class AdapterTransformerEmbedder(PretrainedTransformerEmbedder):
    def __init__(
            self,
            model_name: str,
            *,
            adapter_layers: int = 12,
            adapter_kwargs: Optional[Dict[str, Any]] = None,
            external_param: Union[bool, List[bool]] = False,
            max_length: int = None,
            gradient_checkpointing: Optional[bool] = None,
            tokenizer_kwargs: Optional[Dict[str, Any]] = None,
            transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model_name,
            max_length=max_length,
            train_parameters=False,
            last_layer_only=True,
            gradient_checkpointing=gradient_checkpointing,
            tokenizer_kwargs=tokenizer_kwargs,
            transformer_kwargs=transformer_kwargs
        )
        self.adapters = insert_adapters(
            adapter_layers, adapter_kwargs, external_param, self.transformer_model)
        self.adapter_layers = adapter_layers
        self.adapter_kwargs = adapter_kwargs


def insert_adapters(
        adapter_layers: int, adapter_kwargs: Dict[str, Any],
        external_param: Union[bool, List[bool]], transformer_model: BertModel
) -> nn.ModuleList:
    """
    Initialize adapters, insert them into BERT, and return the adapters. Currently, only supports *BERT structures!

    # Parameters

    adapter_layers : `int`, required.
        Starting from the last layer of BERT, the number of layers where the adapter is inserted.
    adapter_kwargs : `Dict`, required.
        Parameters for initializing the `Adapter`.
    external_param : `Union[bool, List[bool]]`
        Whether the adapter's parameters are left empty for external injection.
    transformer_model : `BertModel`
        Pre-trained model.

    # Returns

    adapters_groups : `nn.ModuleList`, required.
        All inserted adapters, used for binding to the model.
    """
    if not isinstance(transformer_model, (BertModel, ElectraModel, RobertaModel)):
        raise ConfigurationError("Currently, only *BERT structures are supported.")


    if isinstance(external_param, bool):
        param_place = [external_param for _ in range(adapter_layers)]
    elif isinstance(external_param, list):
        param_place = [False for _ in range(adapter_layers)]
        for i, e in enumerate(external_param, 1):
            param_place[-i] = e
    else:
        raise ConfigurationError("wrong type of external_param!")

    adapter_kwargs.update(in_features=transformer_model.config.hidden_size)
    adapters_groups = nn.ModuleList([
        nn.ModuleList([
            Adapter(external_param=param_place[i], **adapter_kwargs),
            Adapter(external_param=param_place[i], **adapter_kwargs)
        ]) for i in range(adapter_layers)
    ])

    for i, adapters in enumerate(adapters_groups, 1):
        layer = transformer_model.encoder.layer[-i]
        layer.output = AdapterBertOutput(layer.output, adapters[0])
        layer.attention.output = AdapterBertOutput(layer.attention.output, adapters[1])

    return adapters_groups
