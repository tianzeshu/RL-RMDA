import os
import heapq
import logging
import torch
import torch.optim as optim
from train import train_util
from .weight_init import weight_init
from .generate_score_file import evaluate_model, calculate_similarity, add_id
from models.RL_IS import policy_selector, optimize_selector


def train_RMDA_CNER_internal( vocab, _ARGS, reader, root_path, train_file, load_model=None):
    train_loader, dev_loader = train_util.build_data_loaders(_ARGS, reader, train_file, _ARGS.valid_file, _ARGS.batch_size)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)
    initial_model_path = None
    best_epoch = 0
    best_f1 = 0
    patience = 0

    epoch_list = []
    for epoch in range(1, _ARGS.num_epochs + 1):
        epoch_list.append(epoch)
        model = train_util.build_model(_ARGS, vocab)

        # Save the initial parameters of the RMDA-CNER model
        if epoch == 1:
            torch.save(model.state_dict(), root_path + f"/origin_model.th")
            initial_model_path = root_path + f"/origin_model.th"

        # load exist model
        if epoch == 1 and load_model is not None:
            model.load_state_dict(torch.load(load_model))

        model = train_util.load_model(root_path, model, epoch)
        trainer = train_util.build_trainer(model, root_path, train_loader, dev_loader, _ARGS.RMDA_CNER_lr, _ARGS.transformer_model_lr, _ARGS.weight_decay, _ARGS.grad_norm, _ARGS.device)
        trainer.train()
        train_util.change_name(epoch, root_path)
        result = train_util.get_result(epoch, root_path)

        # test f1
        if result["validation_f1-measure-overall"] > best_f1:
            best_f1 = result["validation_f1-measure-overall"]
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        # stop training
        if patience >= _ARGS.patience:
            print("Exceeding the patience epochs, stopping training...")
            break
        print(f"Training for epoch {epoch} completed. Currently, the best is epoch {best_epoch} with an F1 score of {best_f1}.")

    if _ARGS.predict:
        print("begin test")
        test_result = train_util.test_model( _ARGS, reader, vocab, best_epoch, root_path)
        logging.info(test_result)

    # Delete parameters from all epochs except the best one and the one before it.
    for epoch in epoch_list:
        if epoch == best_epoch or epoch == best_epoch - 1:
            continue
        train_util.del_files(root_path + f"/{str(epoch)}.th")

    # These are the parameters from the previous epoch of the best-performing epoch.
    # If the first epoch performs the best, then return None.
    if best_epoch - 1 == 0:
        prev_best_epoch_params_path = None
    else:
        prev_best_epoch_params_path = root_path + f"/{str(best_epoch)}.th"
    best_epoch_params_path = root_path + f"/{str(best_epoch)}.th"
    return best_epoch_params_path, prev_best_epoch_params_path, initial_model_path


def generate_score_file(_ARGS, reader, origin_file, predict_file, write_file, write_id_file, load_model):
    # origin_file is the path to the original dataset file
    # predict_file is the path where the model's predictions are stored
    # load_model is the model being loaded
    # write_file is the file to write the training set with scores
    # write_id_file is the file to write the training set with both scores and IDs
    evaluate_model(
        _ARGS,
        reader,
        origin_file=origin_file,
        predict_file=predict_file,
        load_model=load_model,
    )
    # write_file is the final file containing scores
    calculate_similarity(
        worker_num=_ARGS.worker_num,
        predict_file=predict_file,
        write_file=write_file,
        origin_file=origin_file,
    )
    add_id(
        read_file=write_file,
        write_file=write_id_file
    )


# Training the RL-IS requires the parameters from the best-performing epoch of the trained RMDA-CNER,
# as well as the parameters from its previous epoch.
def selector_train(vocab, _ARGS, reader, reader_id, root_path, RMDA_CNER_best_epoch_params_path, RMDA_CNER_initial_model_path,
                   selector_parameter=None, RMDA_CNER_prev_best_epoch_params_path=None):
    best_f1, patience, best_epoch, best_ner_epoch = 0, 0, 0, 0

    # the averaged representation of the removed instances' representations during the previous epoch (R^*)
    r_avg = torch.zeros(1, _ARGS.lstm_hidden_size).to(_ARGS.device)

    RMDA_CNER_model = train_util.build_model(_ARGS, vocab)

    if RMDA_CNER_best_epoch_params_path != None:
        RMDA_CNER_model.load_state_dict(torch.load(RMDA_CNER_best_epoch_params_path))

    train_loader = train_util.build_data_loader(_ARGS, reader_id, _ARGS.train_file, _ARGS.batch_size)
    dev_loader = train_util.build_data_loader(_ARGS, reader, _ARGS.valid_file, _ARGS.batch_size)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    # calculate selection threshold p
    p, dataset_len = train_util.get_p(train_loader)

    selector = policy_selector(input_size=_ARGS.lstm_hidden_size * 2).to(_ARGS.device)
    selector.apply(weight_init)
    selector_optimizer = optim.Adam(filter(lambda p: p.requires_grad, selector.parameters()), lr=0.01)
    selector_criterion = torch.nn.BCELoss(reduction='none')
    if selector_parameter:
        print("Load the pre-trained model for the selector")
        selector.load_state_dict(torch.load(selector_parameter))

    # train RL-IS
    last_f1 = 0
    last_representation_all = []
    last_psi_index = []
    all_f1 = []
    for epoch in range(1, _ARGS.selector_num_epochs + 1):
        cnt = 0
        representation_all = []
        action_all = []
        id_all = []
        id_int_all = []
        for data in train_loader:
            id = [item["id"] for item in data["metadata"]]
            score = [item["score"] for item in data["score"]]
            id_int = [int(item["id"].split("_")[0]) for item in data["metadata"]]
            # get state representation
            with torch.no_grad():
                # get text representation from RMDA_CNER (R_j^k)
                # size [batch_size, sentence_length, representation_size]
                text_representation = RMDA_CNER_model.get_representation(data["tokens"], domain=data["worker"])
                # size [batch_size, representation_size]
                text_representation = torch.mean(text_representation, dim=1)
                r_avg_expand = r_avg.expand_as(text_representation).to(_ARGS.device)
                text_representation = torch.cat([text_representation, r_avg_expand], dim=-1)

                for i in range(len(id)):
                    representation_all.append({"id": id[i], "representation": text_representation[i].to("cpu"), "score": score[i]})

                # get action_point (ap_j)
                selector.eval()
                score = torch.FloatTensor(score).unsqueeze(1).to(_ARGS.device)
                action_point = selector(text_representation, score)
                action_point = action_point.squeeze().cpu().detach().numpy()

            action_all.extend(action_point)
            id_all.extend(id)
            id_int_all.extend(id_int)
            cnt += 1

        min_number = heapq.nsmallest(int(dataset_len * p), action_all)
        min_index = []
        for t in min_number:
            index = action_all.index(t)
            min_index.append(id_all[index])
            action_all[index] = 0
        psi_index = min_index

        train_util.write_select_data(root_path, _ARGS, train_loader, epoch, min_index)

        # Reload the initial parameters of the RMDA_CNER model to train on the filtered dataset and evaluate its F1 score.
        RMDA_CNER_model = train_util.build_model(_ARGS, vocab)
        RMDA_CNER_model.load_state_dict(torch.load(RMDA_CNER_initial_model_path))

        ner_train_loader = train_util.build_data_loader(_ARGS, reader, f"{root_path}/select_data_{epoch}.txt", _ARGS.batch_size)
        ner_dev_loader = train_util.build_data_loader(_ARGS, reader, _ARGS.valid_file, _ARGS.batch_size)
        ner_train_loader.index_with(vocab)
        ner_dev_loader.index_with(vocab)

        if not os.path.exists(f"{root_path}/{epoch}/"):
            os.makedirs(f"{root_path}/{epoch}/")
        ner_patience, ner_best_epoch, ner_best_f1 = 0, 0, 0

        if RMDA_CNER_prev_best_epoch_params_path != None:
            RMDA_CNER_model.load_state_dict(torch.load(RMDA_CNER_prev_best_epoch_params_path))

        epoch_list = []
        for ner_epoch in range(1, _ARGS.selector_ner_train_epochs + 1):
            trainer = train_util.build_trainer(RMDA_CNER_model, f"{root_path}/{epoch}/", ner_train_loader, ner_dev_loader, _ARGS.RMDA_CNER_lr, _ARGS.transformer_model_lr, _ARGS.weight_decay, _ARGS.grad_norm, _ARGS.device)
            trainer.train()
            train_util.change_name(ner_epoch, f"{root_path}/{epoch}/")
            result = train_util.get_result(ner_epoch, f"{root_path}/{epoch}/")
            print(f"Result of the {epoch}th filtering epoch, iteration {ner_epoch} f1: {result['validation_f1-measure-overall']}")

            # test f1
            if result["validation_f1-measure-overall"] > ner_best_f1:
                ner_best_f1 = result["validation_f1-measure-overall"]
                ner_best_epoch = ner_epoch
                ner_patience = 0
            else:
                ner_patience += 1
            epoch_list.append(ner_epoch)
            if ner_patience >= _ARGS.patience:
                print("Exceeding the RMDA_CNER patience epochs, stopping training...")
                break

        f1 = ner_best_f1
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            best_ner_epoch = ner_best_epoch
            patience = 0
        else:
            patience += 1

        if patience >= _ARGS.selector_patience:
            print("Exceeding the RL-IS patience epochs, stopping training...")
            break
        all_f1.append(f1)

        reward = f1 - last_f1
        print(f"This epoch's F1 is {f1}, last epoch's F1 was {last_f1}, reward is {reward}")
        logging.info(f"This epoch's F1 is {f1}, last epoch's F1 was {last_f1}, reward is {reward}")


        last_f1 = f1
        omega = list(set(psi_index) - set(list(set(psi_index).intersection(last_psi_index))))
        last_omega = list(set(last_psi_index) - set(list(set(psi_index).intersection(last_psi_index))))

        omega_sentense_representations = []
        score_list = []
        temp_dict = {}
        if len(omega) != 0:
            for omega_index in omega:
                for item in representation_all:
                    if item.get("id") == omega_index:
                        temp_dict = item
                omega_sentense_representations.append(temp_dict.get("representation"))
                score_list.append([temp_dict.get("score")])
            y_label_in_batch = [0] * len(omega_sentense_representations)
            reward_list = [reward] * len(omega_sentense_representations)
            if epoch != 1:
                optimize_selector(_ARGS, selector, selector_optimizer, selector_criterion, omega_sentense_representations, score_list, y_label_in_batch, reward_list)

        last_omega_sentense_representations = []
        last_score_list = []
        if len(last_omega) != 0:
            for omega_index in last_omega:
                for item in last_representation_all:
                    if item.get("id") == omega_index:
                        temp_dict = item
                last_omega_sentense_representations.append(temp_dict.get("representation"))
                last_score_list.append([temp_dict.get("score")])
            y_label_in_batch = [0] * len(last_omega_sentense_representations)
            reward_list = [-reward] * len(last_omega_sentense_representations)
            if epoch != 1:
                optimize_selector(_ARGS, selector, selector_optimizer, selector_criterion, last_omega_sentense_representations, last_score_list, y_label_in_batch, reward_list)

        last_representation_all = representation_all
        last_psi_index = psi_index

        torch.save(selector.state_dict(), root_path + f"/selector_{str(epoch)}.th")

        psi_sentense_representations = []
        if len(psi_index) != 0:
            for index in psi_index:
                for item in representation_all:
                    if item.get("id") == index:
                        temp_dict = item
                psi_sentense_representations.append(temp_dict.get("representation"))
            r_avg = torch.mean(torch.stack(psi_sentense_representations).squeeze(), dim=0).unsqueeze(dim=0)
            r_avg = r_avg[:, _ARGS.lstm_hidden_size:]

    print("Below are the best F1 scores obtained by training RMDA-CNER on datasets filtered "
          "by RL-IS and evaluated on the validation set:")
    for i in range(1, len(all_f1) + 1):
        print(f"epoch: {i}, f1: {all_f1[i - 1]}")

    if _ARGS.predict:
        print(f"The best epoch during filtering is {best_epoch}")
        test_result = train_util.test_model(_ARGS, reader, vocab, best_ner_epoch, f"{root_path}/{best_epoch}")
        print(test_result)

    return f"{root_path}/select_data_{best_epoch}.txt"
