import re
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from utils import get_case

class DataHandler:

    def __init__(self,
            BERT_tokenizer,
            segment_size,
            punctuations,
            punc_to_class,
            case_to_class,
        ):
        """
        Initializes the DataHandler

        Parameters
        ----------
        tokenizer: transformers.PreTrainedTokenizer
            The BERT's pre-trained tokenizer.
        segment_size: int
            The size of the context window.
        punc_to_class: dict
            A dictionary mapping a punctuation token to the class index.
        case_to_class: dict
            A dictionary mapping a case token to the class index.
        """
        self.tokenizer = BERT_tokenizer
        self.segment_size = segment_size
        self.punctuations = punctuations
        self.punc_to_class = punc_to_class
        self.case_to_class = case_to_class
    
    def extract_tokens_labels(self, sentences):
        """
        Extracts punctuations & cases of every token of the given sentences.

        Parameters
        ----------
        sentences: list(str)
            A list of sentences to be preprocessed.
        
        Returns
        -------
        tokens: list(str)
            A list of tokens tokenized from the given sentences.
        punc_labels: list(int)
            A list of punctuation classes for every token.
        case_labels: list(int)
            A list of case classes for every token.
        """
        tokens, punc_labels, case_labels = [], [], []
        # combine subwords after tokenization
        sentences = [
            " ".join(self.tokenizer.tokenize(sent)).replace(' ##', '')
            for sent in sentences
        ]
        for sent in sentences:
            sent_tokens = sent.split(' ') #tokenize using white-space
            i = 0
            while ( i < len(sent_tokens)):
                if i == len(sent_tokens)-1:
                    curr_token = sent_tokens[i]
                    tokens.append(curr_token.lower())
                    punc_labels.append(0) # index for other 'O'
                    case_labels.append(self.case_to_class[get_case(curr_token)])
                    i += 1
                    continue
                else:
                    curr_token, next_token = sent_tokens[i], sent_tokens[i+1]
                    tokens.append(curr_token.lower())
                    if next_token in self.punc_to_class:
                        punc_label = self.punc_to_class[next_token]
                        i += 1
                        #ignore other consecutive punctuations if found
                        while i < len(sent_tokens) and sent_tokens[i] in self.punc_to_class:
                            i += 1
                    else:
                        punc_label = 0 #label for other 'O
                        i += 1
                    punc_labels.append(punc_label)
                    case_labels.append(self.case_to_class[get_case(curr_token)])
        assert len(tokens) == len(punc_labels) == len(case_labels)
        return tokens, punc_labels, case_labels
    
    def _expand(self, tokens, punc_labels, case_labels):
        """
        Expands the tokens & labels to the sub-token level. Remember that BERT
        tokenizers uses sub-word tokenization.

        Parameters
        ----------
        tokens: list(str)
            A list of tokens.
        punc_labels: list(int)
            A list of punctuation classes, one for each token.
        case_labels: list(int)
            A list of case classes, one for each token.
        
        Returns:
        out_tokens: list(str)
            A list of expanded tokens.
        out_punc_labels: list(int)
            A list of expanded punctuation classes, one for each sub-token.
        out_case_labels: list(int)
            A list of expanded case classes, one for each sub-token.
        """
        out_tokens, out_punc_labels, out_case_labels = [], [], []
        for token, punc, case in zip(tokens, punc_labels, case_labels):
            subwords = self.tokenizer.tokenize(token)
            out_tokens += subwords
            #expand labels with other index (0)
            out_punc_labels += [0]*(len(subwords)-1) + [punc]
            out_case_labels += [0]*(len(subwords)-1) + [case]
        assert len(out_tokens) == len(out_punc_labels) == len(out_case_labels)
        return out_tokens, out_punc_labels, out_case_labels
    
    def _create_samples(self, sub_tokens):
        # Make sure there are enough tokens
        min_required_length = self.segment_size // 2
        if len(sub_tokens) < min_required_length:
            raise ValueError("Input has very short context! " + 
                f"The input sentences must be at least {min_required_length}" +
                "-token long!")
        # convert subtokens in ids
        sub_tokens_ids = [
            self.tokenzier.convert_tokens_to_ids(subtoken)
            for subtoken in sub_tokens
        ]
        #pad the start & end of x
        x_pad = sub_tokens_ids[-((self.segment_size-1)//2-1):] \
                + sub_tokens_ids \
                + sub_tokens_ids[:self.segment_size//2] 
        # divide the sub-tokens into samples based on the segment size
        out_samples = []
        for i in range(len(x_pad)-self.segment_size+2):
            segment = x_pad[i:i+self.segment_size-1]
            #zero at the middle to mark the targeted token
            segment.insert((self.segment_size-1)//2, 0)
            out_samples.append(segment)
        return np.array(out_samples)
    
    def create_train_dataloader(self, sentences, batch_size, shuffle=True):
        """
        Loads the data as a torch.DataLoader for training or validation.

        Parameters
        ----------
        sentences: list(str)
                A list of sentences to be labeled.
        batch_size: int
            The batch size.
        shuffle: bool
            A flat to wheather shuffle the data or not. Shuffling happens to 
            the samples not the words (subwords) or even the sentences.
        
        Returns
        -------
        data_loader: torch.utils.data.DataLoader
            A data loader of the train/valid data.
        """
        # extract tokens & labels
        tokens, punc_labels, case_labels = self.extract_tokens_labels(sentences)
        # expand tokens & labels to the sub-word level
        subtokens, punc_labels, case_labels = self._expand(tokens,
                                                punc_labels, case_labels)
        # create samples
        samples = self._create_samples(subtokens)
        assert len(samples) == len(punc_labels) == len(case_labels)
        # create data loader
        data_set = TensorDataset(
            torch.from_numpy(samples).long(),
            torch.from_numpy(punc_labels).long(),
            torch.from_numpy(case_labels).long(),
        )
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
        return data_loader
      
    def create_test_dataloader(self, sentences, batch_size):
        """
        Loads the data as a torch.DataLoader for testing.

        Parameters
        ----------
        sentences: list(str)
                A list of sentences to be labeled.
        batch_size: int
            The batch size.
        
        Returns
        -------
        data_loader: torch.utils.data.DataLoader
            A data loader of the test data.
        """
        # extract tokens & labels
        tokens, punc_labels, case_labels = self.extract_tokens_labels(sentences)
        # expand tokens & labels to the sub-word level
        subtokens, _, _ = self._expand(tokens, punc_labels, case_labels)
        # create samples
        samples = self._create_samples(subtokens)
        # create data loader without any labels
        data_set = TensorDataset(torch.from_numpy(samples).long())
        data_loader = DataLoader(data_set, batch_size=batch_size)
        return data_loader