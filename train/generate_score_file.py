import torch
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from typing import List
from train import train_util


class SentenceClassifierPredictor(Predictor):
    def predict(self, sentence: List[str]) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)


def evaluate_model(_ARGS, reader, origin_file, predict_file, load_model):
    train_util.seed_everything(_ARGS, _ARGS.seed)
    train_data, dev_data = train_util.read_data(reader, _ARGS.ner_train_file, _ARGS.valid_file)
    vocab = train_util.build_vocab(train_data + dev_data)
    model = train_util.build_model(_ARGS, vocab)
    model.load_state_dict(torch.load(load_model))
    predictor = SentenceClassifierPredictor(model, reader)
    with open(predict_file, "a", encoding="utf-8") as f_w:
        with open(origin_file, "r", encoding="utf-8") as f:
            sentence = []
            for line in f.readlines():
                if line == "\n":
                    tags = predictor.predict(sentence)["tags"]
                    for i in range(len(sentence)):
                        f_w.write(sentence[i])
                        f_w.write(" ")
                        f_w.write(tags[i])
                        f_w.write("\n")
                    f_w.write("\n")
                    sentence = []
                else:
                    line = line.strip()
                    sentence.append(line.split()[0])


def calculate_similarity(worker_num, predict_file, origin_file, write_file):
    with open(write_file, "a", encoding="utf-8") as f_w:
        with open(predict_file, "r", encoding="utf-8") as f_p:
            with open(origin_file, "r", encoding="utf-8") as f_a:
                answers_lines = f_a.readlines()
                predict_lines = f_p.readlines()
                tags_list = []
                predict_tags_list = []
                words_list = []
                score_list = []
                for i in range(len(answers_lines)):
                    answer_line = answers_lines[i]
                    predict_line = predict_lines[i]
                    if answer_line == "\n":
                        # calculate similarity score
                        for j in range(1, worker_num + 1):
                            if tags_list[j][0] == "?":
                                score_list.append(-1)
                                continue
                            new_true_label = 0
                            new_all_label = 0
                            for k in range(len(words_list)):
                                if tags_list[j][k] == predict_tags_list[k] and tags_list[j][k] != 'O':
                                    new_true_label += 1
                                if predict_tags_list[k] != 'O':
                                    new_all_label += 1

                            if new_all_label != 0:
                                new_score = new_true_label * 1.0 / new_all_label
                            else:
                                new_score = 0
                            score = round(new_score, 3)
                            score_list.append(score)

                        for j in range(len(words_list)):
                            f_w.write(words_list[j])
                            f_w.write(" ")
                            for k in range(1, worker_num + 1):
                                f_w.write(tags_list[k][j])
                                if k != worker_num:
                                    f_w.write(" ")
                            f_w.write("\n")
                        f_w.write("- ")
                        for j in range(len(score_list)):
                            f_w.write(str(score_list[j]))
                            if j != worker_num:
                                f_w.write(" ")
                        f_w.write("\n")
                        f_w.write("\n")

                        tags_list = []
                        predict_tags_list = []
                        words_list = []
                        score_list = []
                    else:
                        word = answer_line.strip().split()[0]
                        predict_tag = predict_line.strip().split()[1]
                        words_list.append(word)
                        predict_tags_list.append(predict_tag)
                        for j in range(1, worker_num + 1):
                            tag = answer_line.strip().split()[j]
                            while 1:
                                if len(tags_list) <= j:
                                    tags_list.append([])
                                else:
                                    break
                            tags_list[j].append(tag)


def add_id(read_file, write_file):
    cnt = 1
    with open(write_file, "a", encoding="utf-8") as f1:
        with open(read_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if line == "\n":
                    cnt += 1
                else:
                    line = str(cnt) + " " + line
                f1.write(line)
