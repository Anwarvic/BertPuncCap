from pickle import FALSE
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data

def encode_sentences(sentences, tokenizer):
    """
    Converts words to (BERT) tokens and labels to given encoding.
    Note that words can be composed of multiple tokens.

    Parameters
    ----------
    sentences: list(str)
        A list of strings where each string is a `[word]\t[tag],[case]`.
    tokenizer: transformers.PreTrainedTokenizer
        The BERT's pre-trained tokenizer.

    Returns
    -------
    X: list(int)
        A list of integers where each integer represents the sub-word index
        according to the `tokenizer`.

    Note
    ----
    Since a word can be divided into multiple subwords, the label tag is
    assigned only to the last part.
    """
    out_sentences = []
    for sent in tqdm(sentences, "Preprocessing"):
        subtokens = tokenizer.tokenize(sent.strip())
        x = tokenizer.convert_tokens_to_ids(subtokens) #subword ids
        out_sentences += x
    return out_sentences

def insert_target(x, segment_size):
    """
    Creates segments of surrounding words for each word in x.
    Inserts a zero token halfway the segment to mark the end of the intended
    token.

    Parameters
    ----------
    x: list(int)
        A list of integers representing the whole data as one long encoded
        sentence. Each integer is an encoded subword.
    segment_size: int
        The size of the output samples.
    
    Returns
    -------
    np.array:
        A numpy matrix representing  window of `segment_size` moving over the
        input sample `x`.
    """
    X = []
    #pad the start & end of x
    x_pad = x[-((segment_size-1)//2-1):] + x + x[:segment_size//2] 
    for i in range(len(x_pad)-segment_size+2):
        segment = x_pad[i:i+segment_size-1]
        #zero at the middle to mark the intended token
        segment.insert((segment_size-1)//2, 0)
        X.append(segment)
    return np.array(X)

def preprocess_data(sentences, tokenizer, segment_size):
    """
    Divides the data into samples (X) and labels (Y). Note that one data sample
    can be preprocessed to be multiple based on the `segment_size`.

    Parameters
    ----------
    sentences: list(str)
        A list of sentences to be preprocessed.
    tokenizer: transformers.PreTrainedTokenizer
        The BERT's pre-trained tokenizer.
    encoding_dict: dict
        The dictionary where keys are labels and values are indices.
    segment_size: int
        The size of the sample (X).
    include_punctuations: bool
        A flag for including the case label (default: True).
    include_case: bool
        A flag for including the case label (default: True).
    multiple_labels: bool
        A flag whether to use multiple labels (default: False).
        This is important iff `include_case` and `include_punctuation`
        are `True`.
    
    Returns
    -------
    X: list(int)
        A list of encoded samples.
    """
    X = encode_sentences(sentences, tokenizer)
    X = insert_target(X, segment_size)
    return X

def create_data_loader(X, shuffle, batch_size):
    """
    Converts samples (X) and labels (Y) into TensorDataset.

    Parameters
    ----------
    X: list(int)
        A list of encoded samples.
    Y: list(int)
        A list of encoded labels.
    shuffle: bool
        A flag for shuffling the data.
    batch_size: int
        The batch size.
    
    Returns
    -------
    data_loader: torch.utils.data.DataLoader
        A data loader for the data.
    """
    data_set = TensorDataset(torch.from_numpy(X).long())
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader
