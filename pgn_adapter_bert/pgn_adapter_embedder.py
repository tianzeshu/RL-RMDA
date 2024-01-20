from typing import Optional, Dict, Any
from overrides.overrides import overrides
import torch
import torch.nn as nn
from torch.nn import Parameter
from allennlp.common.checks import ConfigurationError
from .adapter_embedder import AdapterTransformerEmbedder


class PgnAdapterTransformerEmbedder(AdapterTransformerEmbedder):
    def __init__(
            self,
            model_name: str,
            *,
            domain_num: int,
            domain_embedding_dim: int = 8,
            pgn_layers: int = 12,
            share_param: bool = False,
            adapter_layers: int = 12,
            adapter_kwargs: Optional[Dict[str, Any]] = None,
            max_length: int = None,
            gradient_checkpointing: Optional[bool] = None,
            tokenizer_kwargs: Optional[Dict[str, Any]] = None,
            transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model_name,
            adapter_layers=adapter_layers,
            adapter_kwargs=adapter_kwargs,
            external_param=True,
            max_length=max_length,
            gradient_checkpointing=gradient_checkpointing,
            tokenizer_kwargs=tokenizer_kwargs,
            transformer_kwargs=transformer_kwargs
        )

        if pgn_layers > adapter_layers:
            raise ConfigurationError(
                f"pgn_layers {pgn_layers} should less than adapter_layers {adapter_layers}")
        self.pgn_layers = pgn_layers
        self.share_param = share_param

        self.domain_embedding = nn.Embedding(domain_num, domain_embedding_dim)  # max_norm=1.0

        hidden_size = self.transformer_model.config.hidden_size
        adapter_size = adapter_kwargs["adapter_size"]
        size = [2] if share_param else [pgn_layers, 2]
        weights = dict(
            weight_down=nn.Parameter(torch.Tensor(
                *size, adapter_size, hidden_size, domain_embedding_dim)),
            weight_up=nn.Parameter(torch.Tensor(
                *size, hidden_size, adapter_size, domain_embedding_dim))
        )
        if self.adapters[0][0].bias:
            weights.update(
                bias_down=nn.Parameter(torch.zeros(
                    *size, adapter_size, domain_embedding_dim)),
                bias_up=nn.Parameter(torch.zeros(
                    *size, hidden_size, domain_embedding_dim))
            )
        self.weights = nn.ParameterDict(weights)

        self.preset_domain = None
        self.reset_parameters()

        worker_num = domain_num
        exclude_workers = []
        mean_ids = set(i for i in range(1, worker_num)) - set(exclude_workers)
        self.mean_ids = Parameter(torch.tensor(list(mean_ids), dtype=torch.int), requires_grad=False)
        self.worker_embedding_weight = Parameter(torch.full([len(mean_ids)], 1.0 / len(mean_ids)).unsqueeze(0))

    def reset_parameters(self):
        nn.init.normal_(self.weights.weight_down, std=1e-3)
        nn.init.normal_(self.weights.weight_up, std=1e-3)

    @overrides
    def forward(
            self,
            token_ids: torch.LongTensor,
            mask: torch.BoolTensor,
            # if_one_domain: bool,
            type_ids: Optional[torch.LongTensor] = None,
            segment_concat_mask: Optional[torch.BoolTensor] = None,
            sign: Optional[bool] = True,
            domain: Optional[torch.LongTensor] = None,
            # **kwargs,

    ) -> torch.Tensor:
        if (self.training or self.preset_domain is None) and sign:
            if domain is None:
                embedding = self.domain_embedding(self.preset_domain)
            else:
                embedding = self.domain_embedding(domain)
            self.generate_parameters(embedding)
        else:
            embedding = torch.matmul(self.worker_embedding_weight, self.domain_embedding(self.mean_ids))
            self.generate_parameters(embedding)
        return super().forward(token_ids, mask, type_ids=type_ids, segment_concat_mask=segment_concat_mask)

    def generate_parameters(self, embedding: torch.Tensor):
        def batch_matmul(w: torch.Tensor, e):
            ALPHA = "ijklmnopqrstuvwxyz"
            dims = ALPHA[:w.dim() - 1]
            i = 1 if self.share_param else 2
            # ija,ba->ibj
            # ija,ba->ijb
            # ijka,ba->ibjk
            # ijka,ba->ijkb
            return torch.einsum(f"{dims}a,ba->{dims[:i] + 'b' + dims[i:]}", w, e)

        matmul = batch_matmul if embedding.dim() == 2 else torch.matmul
        embedding = embedding.softmax(-1)
        weights = {k: matmul(v, embedding) for k, v in self.weights.items()}

        for i, adapters in enumerate(self.adapters[-self.pgn_layers:]):
            for j, adapter in enumerate(adapters):
                for k, v in weights.items():
                    param = v[j] if self.share_param else v[i, j]
                    setattr(adapter, k, param)
        return