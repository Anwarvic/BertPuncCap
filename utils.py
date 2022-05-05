import re
import os
import yaml
import torch
import warnings
import numpy as np
from collections import OrderedDict

def load_file(filename):
    """reads text file where sentences are separated by newlines."""
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.strip() for line in f.readlines()]
    return data

def parse_yaml(filepath):
    """Parses a yaml file."""
    with open(filepath, "r") as stream:
        return yaml.safe_load(stream)
        
def load_checkpoint(checkpoint_path, device):
    stat_dict = None
    if os.path.exists(os.path.join(checkpoint_path, "best_model")):
        print("Loading best model!")
        stat_dict = torch.load(
            os.path.join(checkpoint_path, 'best_model'),
            map_location=device
        )
    elif os.path.exists(os.path.join(checkpoint_path, "latest_model")):
        print("Loading latest model!")
        stat_dict = torch.load(
                os.path.join(checkpoint_path, 'latest_model'),
                map_location=device
        )
    else:
        warnings.warn("CAUTION! Initializing model from scratch!")
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

def convert_to_full_tokens(subwords, punc_pred, case_pred):
    i = 0
    curr_word = ""
    out_tokens, punc_preds, case_preds = [], [], []
    while( i < len(subwords)):
        curr_word += subwords[i]
        while(i+1 < len(subwords) and subwords[i+1].startswith("##")):
            i += 1
            curr_word += subwords[i][2:]
        out_tokens.append(curr_word)
        punc_preds.append(punc_pred[i])
        case_preds.append(case_pred[i])
        curr_word = ""
        i += 1
    return out_tokens, punc_preds, case_preds

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