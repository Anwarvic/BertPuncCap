import os
import logging

# create logger for the project.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_pretrained_bert_tokenizer(pretrained_name):
    """
    Loads pretrained BERT model and tokenizer from HuggingFace.

    Parameters
    ----------
    pretrained_name: str
        The pre-trained name for the model on HuggingFace's hub.
        Ref: https://huggingface.co/models
    
    Returns
    -------
    tokenizer: transformers.PreTrainedTokenizer
            A tokenizer object from the HuggingFace's `transformers` package.
    model: transformers.PreTrainedModel
        The BERT pre-trained model at HuggingFace's `transformers` package.
    """
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

def load_optimizer(bert_punc_cap, optimizer_name, learning_rate):
    """
    Loads the optimizer from PyTorch.

    Parameters
    ----------
    bert_punc_cap: torch.nn.Module
        The model responsible for restoring punctuations & capitalization.
    optimizer_name: str
        The name of the Optimizer Module on PyTorch.
    learning_rate: float
        The learning rate.

    Returns
    -------
    torch.optim.Optimizer
        The optimizer.
    
    Raises
    ------
    ValueError:
        If the given name wasn't supported!
    """
    if optimizer_name.lower() == "adam":
        from torch.optim import Adam
        return Adam(bert_punc_cap.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"{optimizer_name} optimizer is not supported!")

def load_criterion(criterion_name):
    """
    Loads the criterion class from PyTorch.

    Parameters
    ----------
    criterion_name: str
        The name of the criterion class on PyTorch.

    Returns
    -------
    torch.nn.Loss
        The Loss function.
    
    Raises
    ------
    ValueError:
        If the given name wasn't supported!
    """
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
    """
    Creates train, valid, and test dataloaders for training.

    Parameters
    ----------
    dataset_name: str
        The name of the dataset.
    langs: list(str)
        A list of supported languages.
    tokenizer: transformers.PreTrainedTokenizer
        A tokenizer object from the HuggingFace's `transformers` package.
    segment_size: int
        The
    batch_size: int
        The batch size.
    punc_to_class: dict
        A dictionary mapping a punctuation token to the class index.
    case_to_class: dict
        A dictionary mapping a case token to the class index.
    
    Returns
    -------
    train_dataloader: torch.utils.data.DataLoader
        A data loader of the train data.
    valid_dataloader: torch.utils.data.DataLoader
        A data loader of the valid data.
    test_dataloader: torch.utils.data.DataLoader
        A data loader of the test data.
    """
    from data_handler import DataHandler
    if dataset_name == "mTEDx":
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
    data_handler = \
        DataHandler(tokenizer, segment_size, punc_to_class,case_to_class)
    logging.info("Creating dataloader for train data")
    train_dataloader = \
            data_handler.create_dataloader(train_sents, batch_size, True)
    logging.info("Creating dataloader for valid data")
    valid_dataloader = \
            data_handler.create_dataloader(valid_sents, batch_size, True)
    logging.info("Creating dataloader for test data")
    test_dataloader  = \
            data_handler.create_dataloader(test_sents,  batch_size)
    return train_dataloader, valid_dataloader, test_dataloader



if __name__ == "__main__":
    import argparse
    from pprint import pformat

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234,
                help='Seed for PyTorch, Numpy and random')
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
    parser.add_argument('--stop_metric', type=str,
        choices=["valid_loss", "punc_valid_loss", "case_valid_loss",
                 "punc_overall_f1", "case_overall_f1", "overall_f1"],
        help='The metric at which early-stopping should be applied.')
    # parse arguments
    args = vars(parser.parse_args())

    # other hyper-parameters
    args["punc_to_class"] = {
        ",": 1, "،": 1,
        ".": 2, "...": 2,
        "?": 3, '؟': 3, '¿': 3,
        "!": 4, "¡": 4,
        ":": 5,
        ";": 6, "؛": 6,
    }
    args["class_to_punc"] = {
        0: '',   #'O'
        1: ',',  #'COMMA'
        2: '.',  #'PERIOD'
        3: '?',  #'QUESTION'
        4: '!',  #"EXCLAMATION"
        5: ':',   #"COLON"
        6: ';',  #"SEMICOLON"
    }
    args["case_to_class"] = {
        'O': 0,  #Other 
        'F': 1,  #First_cap 
        'A': 2   #All_cap
    }
    args["class_to_case"] = {
        0: 'O',  #Other
        1: 'F',  #First_cap
        2: 'A'   #All_cap
    }
    # log training arguments
    logging.info("Initialize training with the following arguments:")
    logging.info(pformat(args))

    # setting the seed
    logging.info(f"Setting the seed to: {args['seed']}")
    from utils import set_all_seeds
    set_all_seeds(args["seed"])

    # load pre-trained model & tokenizer
    logging.info(f"Loading pre-trained BERT: {args['pretrained_bert']}")
    BERT_tokenizer, BERT_model = \
                load_pretrained_bert_tokenizer(args["pretrained_bert"])     
    # save model's hyper-parameters in save_path
    os.makedirs(args["save_path"], exist_ok=True)
    from utils import write_yaml
    write_yaml({
        "segment_size": args["segment_size"],
        "dropout": 0.3,
        "punc_to_class": args["punc_to_class"],
        "class_to_punc": args["class_to_punc"],
        "case_to_class": args["case_to_class"],
        "class_to_case": args["class_to_case"]
    }, os.path.join(args["save_path"], "config.yaml"))
    
    # create bert_punc_cap
    logging.info("Loading BertPuncCap")
    from model import BertPuncCap
    bert_punc_cap = BertPuncCap(BERT_model, BERT_tokenizer,
            model_path=os.path.join(args["save_path"]))
    
    # load optimizer
    logging.info(f"Loading optimizer: {args['optimizer']}")
    optimizer = load_optimizer(bert_punc_cap, args["optimizer"], args["lr"])
    
    # load criterion
    logging.info(f"Loading criterion: {args['criterion']}")
    criterion = load_criterion(args["criterion"])

    # load data loaders
    logging.info(f"Loading dataset: {args['dataset']} "
                + f"for langs: [{args['langs']}]")
    train_dataloader, valid_dataloader, _ = \
        create_data_loaders(args["dataset"], args["langs"], BERT_tokenizer,
            args["segment_size"], args["batch_size"],
            args["punc_to_class"], args["case_to_class"])
    
    # create Trainer
    from trainer import Trainer
    trainer = Trainer(bert_punc_cap, optimizer, criterion, train_dataloader,
                    valid_dataloader, args["save_path"], args["batch_size"],
                    args["lr"], args["max_epochs"], args["num_validations"],
                    args["alpha"], args["patience"], args["stop_metric"])
    logging.info("Started training...")
    trainer.train()
