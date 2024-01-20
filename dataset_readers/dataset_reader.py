from typing import Dict, List, Iterable
import itertools
import logging
from overrides import overrides
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, PathOrStr
from allennlp.data.fields import TextField, SequenceLabelField, LabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ""
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":
            return True
        else:
            return False


class CrowdDatasetReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        label_namespace: str = "labels",
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._token_indexers = token_indexers
        self.label_namespace = label_namespace

    @overrides
    def _read(self, file_path: PathOrStr) -> Iterable[Instance]:
        with open(file_path, "r", encoding="utf-8") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # Group lines into sentence chunks based on the divider.
            line_chunks = (
                lines
                for is_divider, lines in itertools.groupby(data_file, _is_divider)
                if not is_divider
            )
            for lines in self.shard_iterable(line_chunks):
                fields = [line.strip().split() for line in lines]
                # unzipping trick returns tuples, but our Fields need lists
                tokens, *tags = [list(field) for field in zip(*fields)]
                if "answers.txt" in file_path:
                    for ins in self.text_to_instances(tokens, tags):
                        yield ins
                else:
                    yield self.text_to_instance(tokens, tags[0])

    def text_to_instances(self, tokens, tag_mat) -> Iterable[Instance]:
        """
        we leave worker id 0 as the expert.
        """
        for i in range(len(tag_mat)):
            worker = i + 1
            # if '?' not in set(tag_mat[i]):
            if len(set(tag_mat[i])) > 1:
                yield self.text_to_instance(tokens, tag_mat[i], worker)

    def text_to_instance(
        self,
        words: List[str],
        tags: List[str] = None,
        worker: int = -1
    ) -> Instance:
        """
        worker == -1 means we don't use annotator information.
        """
        sequence = TextField([Token(w) for w in words], self._token_indexers)
        fields: Dict[str, Field] = {"tokens": sequence}
        fields["metadata"] = MetadataField({"words": words, "tags": tags})
        if tags != None:
            fields["tags"] = SequenceLabelField(tags, sequence, self.label_namespace)
        fields["worker"] = LabelField(worker, "worker", True)
        return Instance(fields)
