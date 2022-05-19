import re
import os
import yaml
import torch
import random
import logging
logging.getLogger()
import numpy as np
from glob import glob
from collections import OrderedDict

def load_file(filename):
    """reads text file where sentences are separated by newlines."""
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.strip() for line in f.readlines()]
    return data

def parse_yaml(filepath):
    """Reads & parse a YAML file."""
    with open(filepath, "r", encoding='utf-8') as fin:
        return yaml.safe_load(fin)

def write_yaml(d, filepath):
    """Writes a dictionary into a YAML file"""
    with open(filepath, 'w', encoding='utf-8') as fout:
        yaml.dump(d, fout, default_flow_style=False, allow_unicode=True)

def load_checkpoint(ckpt_path, device, option="best"):
    """Loads a checkpoint on device"""
    stat_dict = None
    if option == "best":
        logging.info("Loading best model!")
        if os.path.exists(os.path.join(ckpt_path, "best.ckpt")):
            stat_dict = torch.load(
                os.path.join(ckpt_path, 'best.ckpt'),
                map_location=device
            )
        else:
            logging.warn("Couldn't load best checkpoint, backing off to latest")
            option = "latest"
    if option == "latest":
        checkpoints = glob(f"{ckpt_path}/*.ckpt")
        # exclude best.ckpt
        if os.path.exists(f"{ckpt_path}/best.ckpt"):
            checkpoints.remove(f"{ckpt_path}/best.ckpt")
        latest_checkpoint = sort_alphanumeric(checkpoints)[-1]
        logging.info(f"Loading latest checkpoint: {latest_checkpoint}")
        stat_dict = torch.load(
                latest_checkpoint,
                map_location=device
        )
    elif option not in {"latest", "best"}:
        raise FileNotFoundError("Can't load pre-trained checkpoint!")
    # remove unneeded parameters if found
    logging.debug("Removing old BertPuncCap keys if found")
    new_stat_dict = {}
    ignore_keys = {"bn.weight", "bn.bias", "bn.running_mean",
        "bn.running_var", "bn.num_batches_tracked", "fc.weight", "fc.bias"}
    if stat_dict:
        for old_key in stat_dict.keys():
            new_key = old_key.partition('.')[-1]
            if new_key in ignore_keys:
                continue
            new_stat_dict[new_key] = stat_dict[old_key]
    return OrderedDict(new_stat_dict)  

def apply_labels_to_input(
        tokens_count_per_sent,
        total_tokens,
        punc_preds,
        cases_preds,
        class_to_punc,
        case_class
    ):
    i, j = 0, 0
    labeled_sentences = []
    curr_sentence = []
    while(i < len(tokens_count_per_sent) and j < len(total_tokens)):
        if len(curr_sentence) == tokens_count_per_sent[i]:
            labeled_sentences.append(" ".join(curr_sentence))
            curr_sentence = []
            i += 1
        else:
            curr_punc = class_to_punc[punc_preds[j]]
            curr_case = case_class[cases_preds[j]]
            curr_token = total_tokens[j]+' '+curr_punc if curr_punc else total_tokens[j]
            if curr_case == 'O':
                pass # do nothing
            elif curr_case == 'F':
                curr_token = curr_token.capitalize()
            elif curr_case == 'A':
                curr_token = curr_token.upper()
            curr_sentence.append(curr_token)
            j += 1
    if curr_sentence: labeled_sentences.append(" ".join(curr_sentence))
    return labeled_sentences

def get_case(word):
    """
    Detects the case of a given word.
    Parameters
    ----------
    word: str
        A string representing a word.
    Returns
    -------
    str
        A character representing the case of the word.
    """
    if word.isupper():
        return 'A' #ALL_CAPS
    elif word.istitle():
        return 'F' #FIRST_CAP
    else:
        return 'O' #OTHER

def sum_params(model):
    """Sums the weights/parameters of a given model."""
    s = 0
    for p in model.parameters():
        n = p.cpu().data.numpy()
        s += np.sum(n)
    return s

def set_all_seeds(seed):
    random.seed(seed)
    # os.environ('PYTHONHASHSEED') = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def sort_alphanumeric(lst):
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(lst, key=alphanum_key)