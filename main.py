import argparse
from datetime import datetime
import os
from train.train_process import train_RMDA_CNER_internal, selector_train, generate_score_file
from train import train_util
from dataset_readers.dataset_reader import CrowdDatasetReader
from dataset_readers.dataset_reader_id import CrowdDatasetReaderId
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer


def train_RMDA_CNER(_ARGS, reader, vocab):
    max_epoch = -1
    for root, dirs, files in os.walk(f"{_ARGS.name}/RMDA_CNER/"):
        for file in files:
            if file.endswith(".th"):
                try:
                    ner_epoch = int(file.split(".")[0].strip())
                    if ner_epoch > max_epoch:
                        max_epoch = ner_epoch
                except:
                    continue

    if max_epoch != -1 and os.path.exists(f"{_ARGS.name}/RMDA_CNER/origin_model.th") and os.path.exists(f"{_ARGS.name}/RMDA_CNER/{max_epoch}.th") and os.path.exists(f"{_ARGS.name}/RMDA_CNER/{max_epoch - 1}.th"):
        initial_model_path = f"{_ARGS.name}/RMDA_CNER/origin_model.th"
        best_epoch_params_path = f"{_ARGS.name}/RMDA_CNER/{max_epoch}.th"
        prev_best_epoch_params_path = f"{_ARGS.name}/RMDA_CNER/{max_epoch - 1}.th"
        print(f"Loading the existing best RNER_CNER model: {best_epoch_params_path}")
    else:
        best_epoch_params_path, prev_best_epoch_params_path, initial_model_path = train_RMDA_CNER_internal(vocab, _ARGS, reader, root_path=f"{_ARGS.name}/RMDA_CNER", train_file=_ARGS.ner_train_file)

    return best_epoch_params_path, prev_best_epoch_params_path, initial_model_path


def score_dataset(reader, _ARGS, best_epoch_params_path):
    if os.path.exists(f"{_ARGS.name}/scored_data/answers_with_similarity_score.txt") and os.path.exists(f"{_ARGS.name}/scored_data/answers_with_similarity_score_id.txt") and os.path.exists(
            f"{_ARGS.name}/scored_data/predict_result.txt"):
        pass
    else:
        generate_score_file(
            _ARGS, reader,
            origin_file=_ARGS.ner_train_file,
            predict_file=_ARGS.name + "/scored_data/predict_result.txt",
            write_file=_ARGS.name + "/scored_data/answers_with_similarity_score.txt",
            write_id_file=_ARGS.name + "/scored_data/answers_with_similarity_score_id.txt",
            load_model=best_epoch_params_path
        )


def train_RL_IS(_ARGS, reader, reader_id, vocab, initial_model_path, best_epoch_params_path):
    if os.path.exists(f"{_ARGS.name}/selector_pretrain.th"):
        pretrain_path = f"{_ARGS.name}/selector_pretrain.th"
    else:
        pretrain_path = train_util.pretrain_selector(vocab, _ARGS, reader_id, best_epoch_params_path)

    select_data_path = selector_train(
        vocab, _ARGS, reader, reader_id, root_path=f"{_ARGS.name}/RL_IS", RMDA_CNER_initial_model_path=initial_model_path,
        RMDA_CNER_best_epoch_params_path=best_epoch_params_path,
        selector_parameter=pretrain_path,
    )

    return select_data_path


if __name__ == "__main__":
    curr_time = datetime.now()
    timestamp = datetime.strftime(curr_time, '%Y%m%d_%H%M%S')

    _ARG_PARSER = argparse.ArgumentParser(description="Configs.")
    _ARG_PARSER.add_argument('--project_name', '-project', type=str, default="RL-RMDA", help='save name.')
    _ARG_PARSER.add_argument('--run_name', '-name', type=str, default=timestamp, help='save name.')
    _ARG_PARSER.add_argument('--predict', '-p', default=True, action="store_true")
    _ARG_PARSER.add_argument('--bert_name', type=str, default='/root/bert-base-cased/', help='Name or path to your bert model.')
    _ARG_PARSER.add_argument('--train_file', type=str, default=None, help='RL-IS train file path.')
    _ARG_PARSER.add_argument('--ner_train_file', type=str, default='./data/CoNLL03/answers.txt', help='RMDA-CNER train file path.')
    _ARG_PARSER.add_argument('--valid_file', type=str, default='./data/CoNLL03/dev.txt', help='validation file path.')
    _ARG_PARSER.add_argument('--test_file', type=str, default='./data/CoNLL03/test.txt', help='test file path.')
    _ARG_PARSER.add_argument('--selector_pretrain_file', default=None, help='selector pretrain')
    _ARG_PARSER.add_argument('--worker_num', type=int, default=47, help='number of annotators')
    _ARG_PARSER.add_argument('--seed', '-s', type=int, default=22, help='random seed.')
    _ARG_PARSER.add_argument('--patience', type=int, default=5, help='RMDA-CNER patience.')
    _ARG_PARSER.add_argument('--num_epochs', type=int, default=20, help='RMDA-CNER epochs.')
    _ARG_PARSER.add_argument('--selector_patience', type=int, default=10, help='Selector_patience.')
    _ARG_PARSER.add_argument('--selector_num_epochs', type=int, default=10, help='RL-IS training epochs.')
    _ARG_PARSER.add_argument('--selector_ner_train_epochs', type=int, default=20, help='The epochs of RMDA-CNER retrains on RL-IS selected dataset.')
    _ARG_PARSER.add_argument('--pretrain_num_epochs', type=int, default=1000, help='Pretrain epochs.')
    _ARG_PARSER.add_argument('--dropout', type=float, default=0.1, help='Dropout.')
    _ARG_PARSER.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    _ARG_PARSER.add_argument('--transformer_model_lr', type=float, default=1e-5, help='Transformer model learning rate.')
    _ARG_PARSER.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay.')
    _ARG_PARSER.add_argument('--grad_norm', type=int, default=3, help='Grad norm.')
    _ARG_PARSER.add_argument('--adapter_size', type=int, default=64, help='Adapter size.')
    _ARG_PARSER.add_argument('--adapter_layers', type=int, default=12, help='Adapter layers.')
    _ARG_PARSER.add_argument('--lstm_hidden_size', type=int, default=400, help='Lstm hidden size.')
    _ARG_PARSER.add_argument('--device', type=str, default="cuda:0", help='Cpu or gpu device. If use cpu, input cpu. If use gpu, input cuda:[gpu id]')
    _ARG_PARSER.add_argument('--RMDA_CNER_lr', type=float, default=1e-3, help='The learn rate of RMDA_CNER.')
    _ARG_PARSER.add_argument('--RL_IS_lr', type=float, default=0.01, help='The learn rate of instance selector.')

    _ARGS = _ARG_PARSER.parse_args()
    project_name = _ARGS.project_name
    name = _ARGS.run_name
    save_path = "./save/" + project_name + "/" + name
    _ARGS.name = save_path
    _ARGS.train_file = save_path + "/scored_data/answers_with_similarity_score_id.txt"
    train_util.seed_everything(_ARGS, _ARGS.seed)

    if not os.path.exists(f"{_ARGS.name}"):
        os.makedirs(f"{_ARGS.name}")
        os.makedirs(f"{_ARGS.name}/scored_data")
        os.makedirs(f"{_ARGS.name}/RMDA_CNER")
        os.makedirs(f"{_ARGS.name}/RL_IS")


    pretrain_transformer = PretrainedTransformerMismatchedIndexer(model_name=_ARGS.bert_name)
    reader = CrowdDatasetReader(token_indexers={"bert": pretrain_transformer})
    reader_id = CrowdDatasetReaderId(token_indexers={"bert": pretrain_transformer})

    train_data, dev_data = train_util.read_data(reader, _ARGS.ner_train_file, _ARGS.valid_file)
    vocab = train_util.build_vocab(train_data + dev_data)

    # Step 1: train RMDA-CNER model
    best_epoch_params_path, prev_best_epoch_params_path, initial_model_path = train_RMDA_CNER(_ARGS, reader, vocab)

    # Step 2: score dataset based on RMDA-CNER
    score_dataset(reader, _ARGS, best_epoch_params_path)

    # Step 3: build RL-IS
    train_RL_IS(_ARGS, reader, reader_id, vocab, initial_model_path, best_epoch_params_path)
