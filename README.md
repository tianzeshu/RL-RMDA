# RL-RMDA

This code is the official implementation of the paper: Reinforcement-based Denoising for Crowdsourcing Named Entity Recognition in Social Networks.

## 1 Modules
The modules I have implemented are listed below.

1. **dataset_readers**

    The `dataset_readers` module is responsible for reading datasets, which consists of two files. The `dataset_reader.py` file is designed to read the raw dataset and provide data for RMDA-CNER. On the other hand, the `dataset_reader_id.py` file is responsible for reading the scored dataset generated in Step 2 and supplying it for the training of RL-IS in Step 3.
    
2. **models**

    The `models` module contains two models of RL-RMDA: RMDA-CNER and RL-IS. 

3. **pgn_adapter_bert**

    The `pgn_adapter_bert` module lies contains detailed implementation process of PGN-Adapter-BERT. 
    
4. **train**
    
    The train module encompasses the entire training process. Within this module, `train_process.py` embodies the concrete implementation of the training workflow. Additionally, `generate_score_file.py` is responsible for implementing the calculation of similarity scores for the dataset in Step 2.

5. **selector_pretrain**
    
    The `selector_pretrain` module stores the parameters used during the pretraining phase of RL-IS. Initially, this module is empty as it does not contain any pretraining parameters.

## 2 Usage
1. System Requirements

   - BERT Model: [Hugging Face BERT (bert-base-cased)](https://huggingface.co/bert-base-cased)
   - Python: Version 3.8.8
   - Required Libraries: See `requirements.txt` for a list of libraries to be installed.


2. To run the main program, use the following command:
   - `python main.py [OPTIONS]` 
   - Available Options:
   
    | Option| Description                                                                                                                               | Default Value                 |
    |-------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|-------------------------------------------|
    | `--project_name (-project)`| Name for the project.                                                                                                                     | `RL-RMDA`                     |
    | `--run_name (-name)`| Specific name for this run.                                                                                                               | Current timestamp             |
    | `--predict (-p)`| Testing RMDA-CNER using the test dataset.                                                                                                 | `True`                        |
    | `--bert_name`                   | Path to your BERT model.                                                                                                                  | `/root/bert-base-cased/`           |
    | `--train_file`                  | The scored dataset generated in Step 2 is used for RL-IS. If no existing files are present, the initial value is None.                    | None                          |
    | `--ner_train_file`              | RMDA-CNER train file path. For CoNLL03, the input is `./data/CoNLL03/answers.txt`. For FMKKM11, the input is `./data/FMKKM11/answers.txt` | `./data/CoNLL03/answers.txt`  |
    | `--valid_file`                  | Validation file path. For CoNLL03, the input is `./data/CoNLL03/dev.txt`. For FMKKM11, the input is `./data/FMKKM11/dev.txt`              | `./data/CoNLL03/dev.txt`      |
    | `--test_file`                   | Test file path. For CoNLL03, the input is `./data/CoNLL03/test.txt`. For FMKKM11, the input is `./data/FMKKM11/test.txt`                  | `./data/CoNLL03/test.txt`     |
    | `--selector_pretrain_file`      | The pretraining file for RL-IS. If not available, the initial value is set to None.                                                       | None                          |
    | `--worker_num`                  | Number of annotators. For CoNLL03, the input is 47, For FMKKM11, the input is 269.                                                        | 47                            |
    | `--seed (-s)`                   | Random seed, and we provide a reference list: [22, 6622, 6699, 333555, 999111].                                                           | 22                            |
    | `--patience`                    | RMDA-CNER patience.                                                                                                                       | 5                             |
    | `--num_epochs`                  | RMDA-CNER epochs.                                                                                                                         | 20                            |
    | `--selector_patience`           | Selector patience.                                                                                                                        | 10                            |
    | `--selector_num_epochs`         | RL-IS training epochs.                                                                                                                    | 10                            |
    | `--selector_ner_train_epochs`   | Epochs of RMDA-CNER retrains on RL-IS dataset.                                                                                            | 20                            |
    | `--pretrain_num_epochs`         | RL-IS pretrain epochs.                                                                                                                    | 1000                          |
    | `--dropout`                     | Dropout rate.                                                                                                                             | 0.1                           |
    | `--batch_size`                  | Batch size.                                                                                                                               | 64                            |
    | `--transformer_model_lr`        | Transformer model learning rate.                                                                                                          | 1e-5                          |
    | `--weight_decay`                | Weight decay.                                                                                                                             | 0.01                          |
    | `--grad_norm`                   | Grad norm.                                                                                                                                | 3                             |
    | `--adapter_size`                | Adapter size.                                                                                                                             | 64                            |
    | `--adapter_layers`              | Adapter layers.                                                                                                                           | 12                            |
    | `--lstm_hidden_size`            | LSTM hidden size.                                                                                                                         | 400                           |
    | `--device`                      | CPU or GPU device.                                                                                                                        | `cuda:0` (For CPU, use `cpu`) |
    | `--RMDA_CNER_lr`                | Learning rate of RMDA_CNER.                                                                                                               | 1e-3                          |
    | `--RL_IS_lr`                    | Learning rate of the instance selector.                                                                                                   | 0.01                          |

3. Example Usage
   
    To initiate training with a custom BERT path and a distinct project name, you can use:
    `python main.py --bert_name /path_to_your_bert --run_name MySaveName`

4. Outputs

    All output files are saved in the "save" folder located at the root of the project directory. The "save" folder contains the following subfolders and their corresponding output: 

    - **RMDA_CNER**: In the "RMDA_CNER" folder, you can find the training output generated during Step 1 (RMDA-CNER training). This includes model checkpoints, training logs, and other relevant files.

    - **scored_data**: The "scored_data" folder contains the results of Step 2, where the trained RMDA-CNER model is used to score the dataset. This folder includes files that contain scoring results for the dataset based on the trained model's predictions.

    - **RL_IS**: The "RL_IS" folder stores the training parameters and results of RL-IS (Step 3). This folder contains the following subfolders:

      - **Epoch-wise Selection Results:**
        - Each epoch's selected data is saved as "select_data_{epoch}.txt", containing the instances that were selected by RL-IS.
        - Instances that are not selected are saved as "not_select_data_{epoch}.txt".

      - **Re-Training Outputs:**
        - Inside the numbered subfolders, you can find the outputs of retraining the RMDA-CNER model using the dataset selected by RL-IS for each epoch of reinforcement learning.
      
    - The `selector_pretrain.th` stores the parameters used during the pretraining phase of RL-IS.

5. GPU use

    When the batch_size is set to 64, a minimum of 16GB of GPU memory and 32GB of RAM are required.

## 3 References
1. Xin Zhang, Guangwei Xu, Yueheng Sun, Meishan Zhang, Pengjun Xie. "[Crowdsourcing Learning as Domain Adaptation: A Case Study on Named Entity Recognition](https://aclanthology.org/2021.acl-long.432)." Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 2021.
2. Nooralahzadeh Farhad, Lonning Jan Tore, Ovrelid Lilja. "[Reinforcement-based denoising of distantly supervised NER with partial annotation](https://aclanthology.org/D19-6125)." Proceedings of the 2nd Workshop on Deep Learning Approaches for Low-Resource NLP. 2019.
3. Yaosheng Yang, Wenliang Chen, Zhenghua Li, Zhengqiu He and Min Zhang."[Distantly supervised NER with partial annotation learning and reinforcement learning](https://aclanthology.org/C18-1183)." Proceedings of the 27th International Conference on Computational Linguistics. 2018.

## 4 

If you find this work helpful in your research, please kindly consider citing the following paper. The bibtex are listed below:

    @INPROCEEDINGS{Tian2406:Reinforcement,
    AUTHOR="Zeshu Tian and Hongli Zhang and Yan Wang",
    TITLE="Reinforcement-based Denoising for Crowdsourcing Named Entity Recognition in
    Social Networks",
    BOOKTITLE="2024 IEEE International Conference on Communications (ICC): SAC Social
    Networking Track (IEEE ICC'24 - SAC-07 SN Track)",
    ADDRESS="Denver, USA",
    PAGES="6.99",
    DAYS=9,
    MONTH=jun,
    YEAR=2024,
    ABSTRACT="Crowdsourced Named Entity Recognition (CNER) is an effective method in
    social network data mining, enabling the swift identification and
    categorization of vast textual data, thereby enhancing the understanding
    and analysis of user behaviors and communication patterns. However, the
    diversity of such data often leads to inconsistent annotation quality,
    compromising recognition accuracy. We propose a novel framework that
    synergizes reinforcement learning with multi-source domain adaptation to
    bolster the accuracy of social CNER. By filtering out low-quality instances
    and considering annotator reliability in the multi-source domain adaptation
    crowdsourcing method, our framework not only adapts to the varied quality
    of annotations but also mitigates the impact of annotation noise. Validated
    on both the social network crowdsourced benchmark dataset FMKKM11 and the
    general-purpose CoNLL03, our method consistently outperforms
    state-of-the-art methods in terms of model accuracy and noise robustness."
    }
