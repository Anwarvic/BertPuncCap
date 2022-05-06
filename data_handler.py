import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from utils import get_case

class DataHandler:

    def __init__(self, tokenizer, segment_size, punc_to_class, case_to_class):
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
        self.tokenizer = tokenizer
        self.segment_size = segment_size
        self.punc_to_class = punc_to_class
        self.case_to_class = case_to_class
    
    def _extract_tokens_labels(self, sentences, desc="Processing Input"):
        """
        Extracts punctuations & cases of every token of the given sentences.

        Parameters
        ----------
        sentences: list(str)
            A list of sentences to be preprocessed.
        desc: str
            A text describing the task. This method can be used either for
            extracting labels or cleaning (removing punctuation & cases).
        
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
        for sent in tqdm(sentences, desc=desc):
            tmp_tokens, tmp_punc_labels, tmp_case_labels = [], [], []
            sent_tokens = sent.split(' ') #tokenize using white-space
            i = 0
            while ( i < len(sent_tokens)):
                if i == len(sent_tokens)-1:
                    curr_token = sent_tokens[i]
                    # if curr_token is a special token
                    if curr_token in self.tokenizer.special_tokens_map.values():
                        tmp_tokens.append(curr_token)
                        tmp_case_labels.append(0) #index for other case 'O'
                    else: 
                        tmp_tokens.append(curr_token.lower())
                        tmp_case_labels.append(
                            self.case_to_class[get_case(curr_token)]
                        )
                    tmp_punc_labels.append(0) # index for other punctuation 'O'
                    i += 1
                    continue
                else:
                    curr_token, next_token = sent_tokens[i], sent_tokens[i+1]
                    # if curr_token is a special token
                    if curr_token in self.tokenizer.special_tokens_map.values():
                        tmp_tokens.append(curr_token)
                        tmp_case_labels.append(0) #index for other case 'O'
                    else:
                        tmp_tokens.append(curr_token.lower())
                        tmp_case_labels.append(
                            self.case_to_class[get_case(curr_token)]
                        )
                    # if next_token is a punctuation
                    if next_token in self.punc_to_class:
                        punc_label = self.punc_to_class[next_token]
                        i += 1
                        #ignore other consecutive punctuations if found
                        while i < len(sent_tokens) and sent_tokens[i] in self.punc_to_class:
                            i += 1
                    else:
                        punc_label = 0 #label for other punctuation 'O
                        i += 1
                    tmp_punc_labels.append(punc_label)
            assert len(tmp_tokens) == len(tmp_punc_labels) == len(tmp_case_labels), \
                f"Size mismatch when {desc}"
            tokens.append(tmp_tokens)
            punc_labels.append(tmp_punc_labels)
            case_labels.append(tmp_case_labels)
        assert len(tokens) == len(punc_labels) == len(case_labels), \
            f"Size mismatch when {desc}"
        return tokens, punc_labels, case_labels
    
    def _expand(self, tokens, punc_labels, case_labels):
        """
        Expands the tokens & labels to the sub-token level. Remember that BERT
        tokenizers uses sub-word tokenization and one word can be divided into
        multiple 

        Parameters
        ----------
        tokens: list(list(str))
            A list of a list of tokens.
        punc_labels: list(list(int))
            A list of a list of punctuation classes, one for each token.
        case_labels: list(list(int))
            A list of a list of case classes, one for each token.
        
        Returns:
        out_tokens: list(list(str))
            A list of a list of expanded tokens.
        out_punc_labels: list(list(int))
            A list of a list of expanded punctuation classes, one for each
            sub-token.
        out_case_labels: list(list(int))
            A list of a list of expanded case classes, one for each sub-token.
        """
        out_tokens, out_punc_labels, out_case_labels = [], [], []
        for i in range(len(tokens)):
            tmp_tokens, tmp_punc_labels, tmp_case_labels = [], [], []
            for token, punc, case in zip(tokens[i], punc_labels[i], case_labels[i]):
                # don't tokenize special tokens like [unk]
                subwords = self.tokenizer.tokenize(token) \
                    if token != self.tokenizer.special_tokens_map.values() \
                    else [token]
                tmp_tokens += subwords
                #expand labels with other index (0)
                tmp_punc_labels += [0]*(len(subwords)-1) + [punc]
                tmp_case_labels += [0]*(len(subwords)-1) + [case]
            assert len(tmp_tokens) == len(tmp_punc_labels) == len(tmp_case_labels), \
                "Size mismatch when expanding"
            # append results to bigger list
            out_tokens.append(tmp_tokens)
            out_punc_labels.append(tmp_punc_labels)
            out_case_labels.append(tmp_case_labels)
            assert len(out_tokens) == len(out_punc_labels) == len(out_case_labels), \
                "Size mismatch when expanding"
        return out_tokens, out_punc_labels, out_case_labels
    
    def _flatten(self, tokens, punc_labels, case_labels):
        """
        Converts the nested list of tokens to a list of tokens. Same happens
        for punctuations & cases.

        Parameters
        ----------
        tokens: list(list(str))
            A nested list of tokens.
        punc_labels: list(list(int))
            A nested list of punctuation classes.
        case_labels: list(list(int))
            A nested list of cases classes.
        
        Returns
        ------
        out_tokens: list(str)
            A flattened list of the same input tokens.
        out_punc_labels: list(int) 
            A flattened list of the same input punctuation classes.
        out_case_labels:
            A flattened list of the same input case classes.
        """
        out_tokens, out_punc_labels, out_case_labels = [], [], []
        for sent_tokens, sent_puncs, sent_cases in zip(tokens, punc_labels, case_labels):
            out_tokens.extend(sent_tokens)
            out_punc_labels.extend(sent_puncs)
            out_case_labels.extend(sent_cases)
        assert len(out_tokens) == len(out_punc_labels) == len(out_case_labels), \
            "Size mismatch when flattening"
        return out_tokens, out_punc_labels, out_case_labels
    
    def _create_samples(self, sub_tokens):
        """
        Converts tokens to samples; by sample I mean a complete example with
        `segment_size` context that the model can be trained on.
        Every example has a sub-token in the middle with `segment_size/2` words
        on the left and segment_size/2 words on the right.
        Also sub-tokens will be represented by its index in the `tokenizer`
        vocabulary.

        Parameters
        ----------
        sub_tokens: list(str)
            A list of sub-tokens.
        
        Returns
        -------
        out_samples: list(list(int))
            A nested list of samples/examples for the model to be trained on.
            The shape of list should be (num_sub-tokens x segment_size).
        """
        # Make sure there are enough tokens
        min_required_length = self.segment_size // 2
        if len(sub_tokens) < min_required_length:
            raise ValueError("Input has very short context! " + 
                f"The input sentences must be at least {min_required_length}" +
                "-token long!")
        # convert subtokens in ids
        sub_tokens_ids = [
            self.tokenizer.convert_tokens_to_ids(subtoken)
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
        return out_samples
    
    def preprocess(self, sentences):
        """
        Preprocesses the input data and convert them into samples that the
        model can train on.

        Parameters
        ----------
        sentences: list(str)
            A list of sentences to be preprocessed.
        
        Returns
        -------
        samples: list(list(int))
            A nested list of samples/examples for the model to be trained on.
            The shape of list should be (num_sub-tokens x segment_size).
        """
        # extract tokens & labels
        tokens, punc_labels, case_labels = \
                                    self._extract_tokens_labels(sentences)
        # expand tokens & labels to the sub-word level
        subtokens, punc_labels, case_labels = self._expand(tokens,
                                                punc_labels, case_labels)
        # flatten data
        subtokens, punc_labels, case_labels = self._flatten(subtokens,
                                                    punc_labels, case_labels)
        # create samples
        samples = self._create_samples(subtokens)
        assert len(samples) == len(punc_labels) == len(case_labels), \
            "Size mismatch when preprocessing"
        return samples, punc_labels, case_labels

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
        samples, punc_labels, case_labels = self.preprocess(sentences)
        assert len(samples) == len(punc_labels) == len(case_labels)
        # create data loader
        data_set = TensorDataset(
            torch.from_numpy(np.array(samples)).long(),
            torch.from_numpy(np.array(punc_labels)).long(),
            torch.from_numpy(np.array(case_labels)).long(),
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
        samples, _, _ = self.preprocess(sentences)
        # create data loader without any labels
        data_set = TensorDataset(torch.from_numpy(np.array(samples)).long())
        data_loader = DataLoader(data_set, batch_size=batch_size)
        return data_loader


if __name__ == "__main__":
    from transformers import BertTokenizer
    from utils import load_file, parse_yaml

    sentences = load_file("data/mTEDx/fr/test.fr")
    params = parse_yaml("models/mbert_base_cased/config.yaml")
    BERT_name = "bert-base-multilingual-cased"
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_name)
    data_handler = DataHandler(bert_tokenizer, params["segment_size"],
            params["punc_to_class"], params["case_to_class"])
    
    # Should be no AssertionError
    data_handler.create_train_dataloader(sentences, 64)

