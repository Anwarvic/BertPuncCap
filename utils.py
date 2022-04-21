import os
import yaml
import torch
import warnings
from collections import OrderedDict


def parse_yaml(filepath):
    with open(filepath, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise FileExistsError(
                "Can't find the configuration file: config.yaml"
            )

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


def clean(text, punctuations, remove_case=True):
    """remove punctuations and possibly case of a given text."""
    if remove_case: text = text.lower()
    return text.translate(str.maketrans('', '', ''.join(punctuations))).strip()

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
        sentences,
        tokens,
        puncs,
        cases,
        class_to_punc,
        case_class
    ):
    i, j = 0, 0
    labeled_sentences = []
    curr_sentence = []
    while(j < len(tokens)):
        if len(curr_sentence) == len(sentences[i].split(' ')):
            labeled_sentences.append(" ".join(curr_sentence))
            curr_sentence = []
            i += 1
        else:
            curr_punc, curr_case = class_to_punc[puncs[j]], case_class[cases[j]]
            curr_token = tokens[j]+curr_punc
            if curr_case == 'O':
                pass # do nothing
            elif curr_case == 'F':
                curr_token = curr_token.capitalize()
            elif curr_case == 'A':
                curr_token = curr_token.upper()
            curr_sentence.append(curr_token)
            j += 1
    return labeled_sentences


