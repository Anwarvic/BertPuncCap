import os
import yaml
import torch
import string
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

def create_encoding_dict(punctuations, cases):
    encoding_dictionary = {}
    for i, punc in enumerate(punctuations):
        encoding_dictionary[punc] = i
    for i, case in enumerate(cases):
        encoding_dictionary[case] = i
    return encoding_dictionary


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
    return text.translate(str.maketrans('', '', ''.join(punctuations)))