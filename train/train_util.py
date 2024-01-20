import heapq
import json
import os
import random
import numpy as np
from allennlp.training.util import evaluate
from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary
from allennlp.data.samplers import BucketBatchSampler
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.nn.initializers import XavierNormalInitializer, OrthogonalInitializer
from torch import optim
from tqdm import tqdm, trange
import torch
from allennlp.models import Model
from allennlp.training.trainer import Trainer
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from typing import Iterable, List, Tuple
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.training.optimizers import HuggingfaceAdamWOptimizer
from pgn_adapter_bert import TransformerMismatchedEmbedder, PgnAdapterTransformerEmbedder
from models.RMDA_CNER import Pgn_RMDA_CNER
from models.RL_IS import optimize_selector, policy_selector
from .weight_init import *


def read_data(reader: DatasetReader, train_path, valid_path) -> Tuple[List[Instance], List[Instance]]:
    training_data = list(reader.read(train_path))
    validation_data = list(reader.read(valid_path))
    return training_data, validation_data


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    return Vocabulary.from_instances(instances)


def build_data_loaders(_ARGS, reader, train_path, valid_path, batch_size) -> Tuple[DataLoader, DataLoader]:
    if "cpu" in _ARGS.device:
        cuda_device = None
    elif "cuda" in _ARGS.device:
        cuda_device = int(_ARGS.device.split(":")[1])
    else:
        cuda_device = None

    batch_sampler = BucketBatchSampler(batch_size=batch_size, sorting_keys=["tokens"], shuffle=True, padding_noise=0)
    train_loader = MultiProcessDataLoader(
        reader,
        train_path,
        batch_sampler=batch_sampler,
        cuda_device=cuda_device,
    )
    dev_loader = MultiProcessDataLoader(
        reader,
        valid_path,
        batch_sampler=batch_sampler,
        cuda_device=cuda_device,
    )
    return train_loader, dev_loader


def build_data_loader(_ARGS, reader, path, batch_size) -> DataLoader:
    if "cpu" in _ARGS.device:
        cuda_device = -1
    elif "cuda" in _ARGS.device:
        cuda_device = int(_ARGS.device.split(":")[1])
    else:
        cuda_device = None

    batch_sampler = BucketBatchSampler(batch_size=batch_size, sorting_keys=["tokens"], shuffle=True, padding_noise=0)
    data_loader = MultiProcessDataLoader(
        reader,
        path,
        batch_sampler=batch_sampler,
        cuda_device=cuda_device,
    )
    return data_loader


def build_model(
        _ARGS,
        vocab: Vocabulary,
) -> Model:
    adapter_layers = _ARGS.adapter_layers
    adapter_kwargs = {
        "adapter_size": _ARGS.adapter_size,
        "bias": True
    }
    tokenizer_kwargs = {
        "do_lower_case": False
    }
    pgn_adapter_transformer = PgnAdapterTransformerEmbedder(
        model_name=_ARGS.bert_name,
        domain_num=_ARGS.worker_num+1,
        adapter_layers=adapter_layers,
        adapter_kwargs=adapter_kwargs,
        tokenizer_kwargs=tokenizer_kwargs
    )
    text_field_embedder = TransformerMismatchedEmbedder(matched_embedder=pgn_adapter_transformer)
    embedder = BasicTextFieldEmbedder(
        {"bert": text_field_embedder}
    )
    encoder = {
        'hidden_size': _ARGS.lstm_hidden_size,
        'num_layers': 1,
        'type': 'lstm'
    }
    initializer = InitializerApplicator(
        regexes=[
            ("tag_projection_layer.weight", XavierNormalInitializer()),
            ("encoder._module.weight_ih.*", XavierNormalInitializer()),
            ("encoder._module.weight_hh.*", OrthogonalInitializer()),
        ]
    )
    return Pgn_RMDA_CNER(_ARGS, vocab, embedder, encoder, _ARGS.worker_num + 1, dropout=_ARGS.dropout, label_namespace="labels",
                         initializer=initializer).to(_ARGS.device)


def build_trainer(
        model: Model,
        serialization_dir: str,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        lr,
        transformer_lr,
        weight_decay,
        grad_norm,
        cuda_device,
) -> Trainer:
    if "cpu" in cuda_device:
        cuda_device = -1
    elif "cuda" in cuda_device:
        cuda_device = int(cuda_device.split(":")[1])
    else:
        cuda_device = None

    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = HuggingfaceAdamWOptimizer(
        model_parameters=parameters,
        parameter_groups=[([".*transformer_model.*"], {"lr": transformer_lr})],
        lr=lr,
        weight_decay=weight_decay
    )
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        patience=5,
        num_epochs=1,
        optimizer=optimizer,
        grad_norm=grad_norm,
        num_gradient_accumulation_steps=1,
        cuda_device=cuda_device
    )
    return trainer


def load_model(path, model, epoch):
    if os.path.exists(path + f"/{str(epoch - 1)}.th"):
        model.load_state_dict(torch.load(path + f"/{str(epoch - 1)}.th"))
    else:
        pass
    return model


def change_name(epoch, path):
    os.rename(path + "/best.th", path + f"/{str(epoch)}.th")
    os.rename(path + "/metrics_epoch_0.json", path + f"/metrics_epoch_{str(epoch)}.json")
    with open(path + f"/metrics_epoch_{str(epoch)}.json", encoding="utf-8") as f:
        dic = json.load(f)
        dic.pop("best_epoch")
        dic.pop("best_validation_precision-overall")
        dic.pop("best_validation_recall-overall")
        dic.pop("best_validation_loss")
        dic["epoch"] = epoch
    with open(path + f"/metrics_epoch_{str(epoch)}.json", "w") as w:
        json.dump(dic, w, ensure_ascii=False)

def get_result(epoch, path):
    with open(path + f"/metrics_epoch_{str(epoch)}.json") as f:
        result_dict = json.load(f)
    return result_dict


def seed_everything(_ARGS, seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available() and _ARGS.device != "cpu":
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def test_model(_ARGS, reader, vocab, best_epoch, name):
    batch_sampler = BucketBatchSampler(batch_size=_ARGS.batch_size, sorting_keys=["tokens"], shuffle=True, padding_noise=0)
    data_loader = MultiProcessDataLoader(
        reader,
        _ARGS.test_file,
        batch_sampler=batch_sampler
    )
    data_loader.index_with(vocab)
    model = build_model(_ARGS, vocab)
    load_file = name + f"/{best_epoch}.th"
    model.load_state_dict(torch.load(load_file))
    if "cpu" in _ARGS.device:
        cuda_device = -1
    elif "cuda" in _ARGS.device:
        cuda_device = int(_ARGS.device.split(":")[1])
    else:
        cuda_device = -1
    results = evaluate(model, data_loader, cuda_device=cuda_device)
    print(results)
    return results



def del_files(dir_path):
    if os.path.isfile(dir_path):
        try:
            os.remove(dir_path)
        except BaseException as e:
            print(e)
    elif os.path.isdir(dir_path):
        file_lis = os.listdir(dir_path)
        for file_name in file_lis:
            tf = os.path.join(dir_path, file_name)
            del_files(tf)


def get_min_p_score_idlist(data_loader, p):
    all_id = []
    all_score = []
    for data in data_loader:
        for index in range(len(data["metadata"])):
            id = data["metadata"][index]["id"]
            score = data["score"][index]["score"]
            all_id.append(id)
            all_score.append(score)
    min_index = []
    min_number = heapq.nsmallest(int(len(all_score) * p), all_score)
    for t in min_number:
        index = all_score.index(t)
        min_index.append(all_id[index])
        all_score[index] = 0
    return min_index


def write_select_data(root_path, _ARGS, data_loader, epoch, min_index):
    cnt = 0
    with open(f"{root_path}/select_data_{epoch}.txt", "a", encoding="utf-8") as f:
        with open(f"{root_path}/not_select_data_{epoch}.txt", "a", encoding="utf-8") as f1:
            for data in data_loader:
                for index in range(len(data["metadata"])):
                    words = data['metadata'][index]["words"]
                    tags = data['metadata'][index]["tags"]
                    id = data["metadata"][index]["id"]
                    worker = data["worker"].cpu().numpy()[index]
                    if id not in min_index:
                        for i in range(len(words)):
                            f.write(words[i])
                            f.write(" ")
                            for j in range(_ARGS.worker_num):
                                if j == worker - 1:
                                    f.write(tags[i])
                                else:
                                    f.write("?")
                                if j != _ARGS.worker_num - 1:
                                    f.write(" ")
                            f.write("\n")
                        f.write("\n")
                    else:
                        for i in range(len(words)):
                            f1.write(words[i])
                            f1.write(" ")
                            for j in range(_ARGS.worker_num):
                                if j == worker - 1:
                                    f1.write(tags[i])
                                else:
                                    f1.write("?")
                                if j != _ARGS.worker_num - 1:
                                    f1.write(" ")
                            f1.write("\n")
                        f1.write("\n")
                    cnt += 1
    return cnt



def get_p(train_loader):
    all_id = []
    all_score = []
    cnt = 0
    for data in train_loader:
        for index in range(len(data["metadata"])):
            id = data["metadata"][index]["id"]
            score = data["score"][index]["score"]
            all_id.append(id)
            all_score.append(score)
            if score < 0.3:
                cnt += 1
    p = cnt * 1.0 / len(all_score)
    return p, len(all_score)


def pretrain_selector(vocab, _ARGS, reader_id, best_ner_parameter):
    train_loader = build_data_loader(_ARGS, reader_id, _ARGS.train_file, _ARGS.batch_size)
    train_loader.index_with(vocab)
    s_avg = torch.zeros(1, _ARGS.lstm_hidden_size).to(_ARGS.device)
    selector = policy_selector(input_size=2 * _ARGS.lstm_hidden_size).to(_ARGS.device)
    selector.apply(weight_init)
    selector_optimizer = optim.Adam(filter(lambda p: p.requires_grad, selector.parameters()), lr=_ARGS.RL_IS_lr)
    selector_criterion = torch.nn.BCELoss(reduction='none')

    model = build_model(_ARGS, vocab)

    if best_ner_parameter != None:
        model.load_state_dict(torch.load(best_ner_parameter))

    sentence_representation_list = []
    score_list = []
    true_label_list = []
    for data in tqdm(train_loader):
        this_representation = model.get_representation(data["tokens"], domain=data["worker"])
        representation = torch.mean(this_representation, dim=1)
        s_avg_expand = s_avg.expand_as(representation)
        representation = torch.cat([representation, s_avg_expand], dim=-1)
        for index in range(len(data["metadata"])):
            score = data["score"][index]["score"]
            sentence_representation_list.append(representation[index].to("cpu"))
            score_list.append([score])

            if score < 0.3:
                true_label_list.append(0)
            else:
                true_label_list.append(1)
    reward_list = [1] * len(true_label_list)
    for epoch in trange(1, _ARGS.pretrain_num_epochs + 1):
        optimize_selector(_ARGS, selector, selector_optimizer, selector_criterion,
                          sentence_representation_list, score_list, true_label_list, reward_list)

    torch.save(selector.state_dict(), _ARGS.name + f"/selector_pretrain.th")
    return _ARGS.name + f"/selector_pretrain.th"


