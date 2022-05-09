from lib2to3.pgen2 import token
import os


def load_pretrained_bert_tokenizer(pretrained_name):
    if pretrained_name in {"bert-base-cased", "bert-base-uncased",
                            "bert-base-multilingual-cased",
                            "bert-base-multilingual-uncased"}:
        from transformers import BertTokenizer, BertModel
        BERT_tokenizer = BertTokenizer
        BERT_module = BertModel
    elif pretrained_name in {"camembert-base"}:
        from transformers import CamembertForMaskedLM, CamembertTokenizer
        BERT_tokenizer = CamembertTokenizer
        BERT_module = CamembertForMaskedLM
    elif pretrained_name in {"flaubert/flaubert_base_cased"}:
        from transformers import FlaubertModel, FlaubertTokenizer
        BERT_tokenizer = FlaubertTokenizer
        BERT_module = FlaubertModel
    else:
        raise ValueError(f"{pretrained_name} model is not supported!")
    # create the objects and return them
    tokenizer = BERT_tokenizer.from_pretrained(pretrained_name)
    model = BERT_module.from_pretrained(pretrained_name)
    return tokenizer, model

def load_optimizer(bert_punc_cap, optimzier_name, learning_rate):
    if optimzier_name.lower() == "adam":
        from torch.optim import Adam
        return Adam(bert_punc_cap.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"{optimzier_name} optimizer is not supported!")

def load_criterion(criterion_name):
    if criterion_name.lower() == "cross_entropy":
        from torch.nn import CrossEntropyLoss
        return CrossEntropyLoss()
    else:
        raise ValueError(f"{criterion_name} criterion is not supported!")

def create_data_loaders(
        dataset_name,
        langs,
        tokenizer,
        segment_size,
        batch_size,
        punc_to_class,
        case_to_class,
    ):
    from data_handler import DataHandler
    if dataset_name.lower() == "mtedx":
        langs = [lang.strip() for lang in langs.split(',')]
        train_sents, valid_sents, test_sents = [], [], []
        for lang in langs:
            base_path = os.path.join("data", dataset_name, lang)
            with open(os.path.join(base_path, "train."+lang)) as fin:
                train_sents.extend(fin.readlines())
            with open(os.path.join(base_path, "valid."+lang)) as fin:
                valid_sents.extend(fin.readlines())
            with open(os.path.join(base_path, "test."+lang)) as fin:
                test_sents.extend(fin.readlines())
    else:
        raise ValueError(f"{dataset_name} dataset is not supported!")
    # create data loaders
    data_handler = DataHandler(tokenizer, segment_size,
        punc_to_class, case_to_class, batch_size)
    train_dataloader = data_handler.create_dataloader(train_sents, batch_size)
    valid_dataloader = data_handler.create_dataloader(valid_sents, batch_size)
    test_dataloader  = data_handler.create_dataloader(test_sents,  batch_size)
    return train_dataloader, valid_dataloader, test_dataloader





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_bert', type=str,
                default="bert-base-multilingual-cased",
                help='A text describing the pretrianed bert from HuggingFace.')
    parser.add_argument('--optimizer', type=str,
                help='A text describing the optimizer to be used.')
    parser.add_argument('--criterion', type=str,
                help='A text describing the optimizer to be used.')
    parser.add_argument('--alpha', type=float, default=0.5,
        help='A float for the tuning parameter between punc_loss & cap_loss.')
    parser.add_argument('--dataset', type=str, default="mTEDx",
                help='A text describing the dataset to be used.')
    parser.add_argument('--langs', type=str, default="fr",
        help='Comma-separated text determining the languages for training.')
    parser.add_argument('--save_path', type=str,
        help='A relative/absolute path to load/create the trained model.')
    parser.add_argument('--batch_size', type=int, default=256,
                help='An integer describing the batch size used for training.')
    parser.add_argument('--segment_size', type=int, default=32,
                help='An integer describing the context size for the model.')
    parser.add_argument('--dropout', type=float, default=0.3,
                help='A float describing the training dropout rate.')
    parser.add_argument('--lr', type=float, default=1e-5,
                help='A float describing the training learning rate.')
    parser.add_argument('--max_epochs', type=int, default=50,
                help='The maximum number of epochs to train the model.')
    parser.add_argument('--num_validations', type=int, default=1,
            help='An integer describing how many times to validate per epoch.')
    parser.add_argument('--patience', type=int, default=5,
        help='An integer of how many validations to wait before early stopping.')
    
    args = parser.parse_args()

    # other hyper-parameters
    args.punc_to_class = {
        ",": 1, "،": 1,
        ".": 2, "...": 2,
        "?": 3, '؟': 3, '¿': 3,
        "!": 4, "¡": 4,
        ":": 5,
        ";": 6, "؛": 6,
    }
    args.class_to_punc = {
        0: '',   #'O'
        1: ',',  #'COMMA'
        2: '.',  #'PERIOD'
        3: '?',  #'QUESTION'
        4: '!',  #"EXCLAMATION"
        5: ':',   #"COLON"
        6: ';',  #"SEMICOLON"
    }
    args.case_to_class = {
        'O': 0,  #Other 
        'F': 1,  #First_cap 
        'A': 2   #All_cap
    }
    args.class_to_case = {
        0: 'O',  #Other
        1: 'F',  #First_cap
        2: 'A'   #All_cap
    }
    print(args)

    # load pre-trained model & tokenizer
    tokenizer, model = load_pretrained_bert_tokenizer(args["pretrained_bert"])
    # load optimzier
    optimzier = load_optimizer(args["optimizer"], args["lr"])
    # load criterion
    criterion = load_criterion(args["criterion"])
    # load data loaders
    train_dataloader, valid_dataloader, test_dataloader = \
        create_data_loaders(args["dataset"], args["langs"], tokenizer,
            args["segment_size"], args["batch_size"],
            args["punc_to_class"], args["case_to_class"])

    