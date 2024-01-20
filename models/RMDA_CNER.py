from typing import Dict, Optional, List, Any, cast, Set
from overrides import overrides
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import SpanBasedF1Measure
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from pgn_adapter_bert import construct_from_params, PgnAdapterTransformerEmbedder


class RMDA_CNER(Model):
    """
    # Parameters
    _ARGS : required
        The settings of the model.
    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : `TextFieldEmbedder`, required
        Used to embed the tokens `TextField` we get as input to the model.
    encoder : `Seq2SeqEncoder`
        The encoder that we will use in between embedding tokens and predicting output tags.
    label_namespace : `str`, optional (default=`labels`)
        This is needed to compute the SpanBasedF1Measure metric.
        Unless you did something unusual, the default value should be what you want.
    label_encoding : `str`, optional (default=`None`)
        Label encoding to use when calculating span f1 and constraining
        the CRF at decoding time . Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if `calculate_span_f1` or `constrain_crf_decoding` is true.
    include_start_end_transitions : `bool`, optional (default=`True`)
        Whether to include start and end transition parameters in the CRF.
    constrain_crf_decoding : `bool`, optional (default=`True`)
        If `True`, the CRF is constrained at decoding time to
        produce valid sequences of tags. If this is `True`, then
        `label_encoding` is required. If `None` and
        label_encoding is specified, this is set to `True`.
        If `None` and label_encoding is not specified, it defaults
        to `False`.
    dropout:  `float`, optional (default=`None`)
        Dropout probability.
    verbose_metrics : `bool`, optional (default = `False`)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    """

    def __init__(
            self,
            _ARGS,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            encoder: Dict[str, Any],
            worker_num: int,
            label_namespace: str = "labels",
            label_encoding: str = "BIO",
            include_start_end_transitions: bool = True,
            constrain_crf_decoding: bool = True,
            dropout: Optional[float] = None,
            verbose_metrics: bool = False,
            initializer: InitializerApplicator = InitializerApplicator(),
            **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        self.encoder = construct_from_params(Seq2SeqEncoder, input_size=text_field_embedder.get_output_dim(), **encoder)
        # self.encoder = encoder
        self._verbose_metrics = verbose_metrics
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self.tag_projection_layer = torch.nn.Linear(self.encoder.get_output_dim(), self.num_tags)

        self.label_encoding = label_encoding
        if constrain_crf_decoding:
            if not label_encoding:
                raise ConfigurationError(
                    "constrain_crf_decoding is True, but no label_encoding was specified."
                )
            labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
            constraints = allowed_transitions(label_encoding, labels)
        else:
            constraints = None

        self.crf = ConditionalRandomField(
            self.num_tags, constraints, include_start_end_transitions=include_start_end_transitions
        )

        self._f1_metric = SpanBasedF1Measure(
            vocab, tag_namespace=label_namespace, label_encoding=label_encoding
        )

        self.embedding_size = _ARGS.lstm_hidden_size
        self.kernel_size = 2
        self.worker_num = worker_num
        self.conv = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.embedding_size,
                              kernel_size=self.kernel_size)
        self.pool1d = nn.MaxPool1d(kernel_size=5)
        self.worker_linear = nn.Linear(in_features=self.embedding_size, out_features=self.worker_num)
        self.cel = nn.CrossEntropyLoss()
        self.pdist = nn.PairwiseDistance(p=2)
        self.dist1_linear = nn.Linear(in_features=text_field_embedder.get_output_dim(), out_features=1)
        self.dist2_linear = nn.Linear(in_features=text_field_embedder.get_output_dim(), out_features=1)
        initializer(self)

    @overrides
    def forward(
            self,  # type: ignore
            tokens: TextFieldTensors,
            tags: torch.LongTensor = None,
            metadata: List[Dict[str, Any]] = None,
            **kwargs,  # to allow for a more general dataset reader that passes args we don't need
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters
        tokens : `TextFieldTensors`, required
            The output of `TextField.as_array()`, which should typically be passed directly to a
            `TextFieldEmbedder`. This output is a dictionary mapping keys to `TokenIndexer`
            tensors.  At its most basic, using a `SingleIdTokenIndexer` this is : `{"tokens":
            Tensor(batch_size, num_tokens)}`. This dictionary will have the same keys as were used
            for the `TokenIndexers` when you created the `TextField` representing your
            sequence.  The dictionary is designed to be passed directly to a `TextFieldEmbedder`,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer gold class labels of shape
            `(batch_size, num_tokens)`.
        metadata : `List[Dict[str, Any]]`, optional, (default = `None`)
            metadata containg the original words in the sentence to be tagged under a 'words' key.
        # Returns
        An output dictionary consisting of:
        logits : `torch.FloatTensor`
            The logits that are the output of the `tag_projection_layer`
        mask : `torch.BoolTensor`
            The text field mask for the input tokens
        tags : `List[List[int]]`
            The predicted tags using the Viterbi algorithm.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised. Only computed if gold label `tags` are provided.
        """
        # text representation layer
        # expert-aware text representation
        pseudo_text_input = self.text_field_embedder(tokens, sign=False, **kwargs)
        # annotator-aware text representation
        embedded_text_input = self.text_field_embedder(tokens, sign=True, **kwargs)
        pseudo_text_input_p = pseudo_text_input.permute(0, 2, 1)
        embedded_text_input_p = embedded_text_input.permute(0, 2, 1)

        # distance layer
        l2_dist = self.pdist(pseudo_text_input_p, embedded_text_input_p)
        l2_dist = torch.mean(l2_dist, 1).squeeze()
        l2_dist = torch.mean(l2_dist).squeeze()

        mask = util.get_text_field_mask(tokens)
        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, mask)
        recon_loss = None

        # reconstruction layer
        if kwargs['domain'][0] != -1 and embedded_text_input.size()[1] > 2:
            cnn_input = encoded_text.permute(0, 2, 1)
            cnn_feats = self.conv(cnn_input)
            cnn_feats = F.tanh(cnn_feats)
            pool1d_value = F.max_pool1d(cnn_feats, cnn_feats.size(2)).squeeze(2)
            worker_feats = self.worker_linear(pool1d_value)
            recon_loss = self.cel(worker_feats, kwargs['domain'])

        if self.dropout:
            encoded_text = self.dropout(encoded_text)
        logits = self.tag_projection_layer(encoded_text)
        best_paths = self.crf.viterbi_tags(logits, mask)

        # Just get the top tags and ignore the scores.
        predicted_tags = cast(List[List[int]], [x[0] for x in best_paths])

        output = {"logits": logits, "mask": mask, "tags": predicted_tags}

        if tags is not None:
            log_likelihood = self.crf(logits, tags, mask)
            if recon_loss:
                output["loss"] = -log_likelihood + recon_loss + l2_dist
            else:
                output["loss"] = -log_likelihood + l2_dist

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.0
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1
            self._f1_metric(class_probabilities, tags, mask)

        if self.training:
            k = self.get_metrics()
            output.update(k)

        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]
        return output

    # text representation for RL-IS state representation
    def get_representation(self, tokens, **kwargs):
        with torch.no_grad():

            embedded_text_input = self.text_field_embedder(tokens, sign=True, **kwargs)
            mask = util.get_text_field_mask(tokens)
            if self.dropout:
                embedded_text_input = self.dropout(embedded_text_input)
            encoded_text = self.encoder(embedded_text_input, mask)
        return encoded_text

    @overrides
    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        `output_dict["tags"]` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """

        def decode_tags(tags):
            return [
                self.vocab.get_token_from_index(tag, namespace=self.label_namespace) for tag in tags
            ]

        output_dict["tags"] = [decode_tags(t) for t in output_dict["tags"]]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_dict = self._f1_metric.get_metric(reset=reset)
        if self._verbose_metrics:
            return f1_dict
        else:
            k = {x: y for x, y in f1_dict.items() if "overall" in x}
            return k

    default_predictor = "sentence_tagger"


class Pgn_RMDA_CNER(RMDA_CNER):
    """
    """

    def __init__(
            self,
            _ARGS,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            # encoder: Dict[str, Any],
            encoder: Any,
            worker_num: int,
            supervised: bool = False,
            crowd_test: bool = False,
            exclude_workers: Set[int] = (0,),
            worker: Optional[int] = None,
            **kwargs
    ) -> None:
        super().__init__(_ARGS, vocab, text_field_embedder, encoder, worker_num, **kwargs)

        self.supervised = supervised
        self.crowd_test = crowd_test
        self.worker = worker
        self.worker_num = worker_num
        if not isinstance(exclude_workers, set):
            exclude_workers = set(exclude_workers)
        if not supervised:
            mean_ids = set(i for i in range(1, worker_num)) - set(exclude_workers)
            self.mean_ids = Parameter(torch.tensor(list(mean_ids), dtype=torch.int), requires_grad=False)

    def pgn_adapter_bert(self) -> PgnAdapterTransformerEmbedder:
        return self.text_field_embedder._token_embedders['bert'].matched_embedder

    def forward(self, worker: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        if self.training or self.crowd_test:
            self.pgn_adapter_bert().preset_domain = None
        else:
            domain_embedding = self.pgn_adapter_bert().domain_embedding
            if isinstance(self.worker, int):
                embedding = domain_embedding.weight[self.worker]
            elif self.supervised:
                embedding = domain_embedding.weight[0]
            else:
                embedding = torch.matmul(self.pgn_adapter_bert().worker_embedding_weight, domain_embedding(self.pgn_adapter_bert().mean_ids))
            self.pgn_adapter_bert().preset_domain = True
            self.pgn_adapter_bert().generate_parameters(embedding)
        return super().forward(domain=worker, worker_num=self.worker_num, **kwargs)
