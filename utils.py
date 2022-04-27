import re
import os
import yaml
import torch
import warnings
from collections import OrderedDict


def parse_yaml(filepath):
    """Parses a yaml file."""
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


def clean(sentences, punctuations, remove_case=True):
    """remove punctuations and possibly case from a given list of sentences."""
    punctuations = ''.join(punctuations)
    cleaned_sentences = []
    for text in sentences:
        if remove_case: text = text.lower()
        cleaned_text = re.sub(punctuations, '', text) # remove punctuations
        cleaned_text = re.sub('\s+', ' ', cleaned_text) # remove multiple spaces
        # cleaned_text = text.translate(
        #     str.maketrans(' ', ' ', punctuations)
        # ).strip()
        cleaned_sentences.append(cleaned_text)
    return cleaned_sentences

def tokenize(sentences, tokenizer):
    """
    Tokenize a list of sentences.

    Parameters
    ----------
    sentences : list(str)
        List of cleaned sentences (without punctuations and possibly cases) to
        be tokenized.
    tokenizer : transformers.PreTrainedTokenizer
        A tokenizer object from the HuggingFace's `transformers` package.
    
    Returns
    -------
    list(str):
        List of tokenized sentences. Tokens are separated by a white space.
    """
    return [
        " ".join(tokenizer.tokenize(sent)).replace(' ##', '')
        for sent in sentences
    ]

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

def extract_punc_case(sentences, tokenizer, punc_to_class, case_to_class):
    tokens, punc_labels, case_labels = [], [], []
    # tokenize sentences
    tokenized_sentences = tokenize(sentences, tokenizer)
    for sent in tokenized_sentences:
        sent_tokens = sent.split(' ')
        i = 0
        while ( i < len(sent_tokens)):
            if i == len(sent_tokens)-1:
                curr_token = sent_tokens[i]
                tokens.append(curr_token.lower())
                punc_labels.append(0) # index for other 'O'
                case_labels.append(case_to_class[get_case(curr_token)])
                i += 1
                continue
            else:
                curr_token, next_token = sent_tokens[i], sent_tokens[i+1]
                tokens.append(curr_token.lower())
                if next_token in punc_to_class:
                    punc_label = punc_to_class[next_token]
                    i += 1
                    #ignore other consecutive punctuations if found
                    while i < len(sent_tokens) and sent_tokens[i] in punc_to_class:
                        i += 1
                else:
                    punc_label = 0 #label for other 'O
                    i += 1
                punc_labels.append(punc_label)
                case_labels.append(case_to_class[get_case(curr_token)])
    assert len(tokens) == len(punc_labels) == len(case_labels)
    return tokens, punc_labels, case_labels

