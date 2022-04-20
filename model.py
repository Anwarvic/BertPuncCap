import os
import torch
import numpy as np
from torch import nn
from tqdm import tqdm

from data import *
from utils import *


class BertPuncCap(nn.Module):
    def __init__(self, BERT_model, BERT_tokenizer, checkpoint_path=''):
        """
        BERT_model: transformers.PreTrainedModel
            The BERT pre-trained model at HuggingFace's `transformers` package.
        BERT_tokenizer: transformers.PreTrainedTokenizer
            A tokenizer object from the HuggingFace's `transformers` package.
        checkpoint_path: str
            The relative/absolute path for the trained model (default: '').
        """
        super(BertPuncCap, self).__init__()
        # save important params
        self.bert = BERT_model
        self.bert.config.output_hidden_states=True
        self.tokenizer = BERT_tokenizer
        self.hparams = parse_yaml(os.path.join(checkpoint_path, "config.yaml"))
        self.punc_dict = self.hparams["punc_dict"]
        self.encoding_dict = create_encoding_dict(
            self.hparams["punc_class"], self.hparams["case_class"]
        )
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # get needed params
        dropout_rate = self.hparams["dropout"]
        hidden_size = self.bert.config.hidden_size
        segment_size = self.hparams["segment_size"]
        punc_size = len(self.hparams["punc_class"])
        case_size = len(self.hparams["case_class"])
        # build one extra layer
        self.punc_bn = nn.BatchNorm1d(segment_size*hidden_size)
        self.punc_fc = nn.Linear(segment_size*hidden_size, punc_size)
        self.case_bn = nn.BatchNorm1d(segment_size*hidden_size)
        self.case_fc = nn.Linear(segment_size*hidden_size, case_size)
        self.dropout = nn.Dropout(dropout_rate)
        # load trained model's stat_dict
        self.load_state_dict(load_checkpoint(checkpoint_path, self.device))


    def forward(self, x):
        x = self.bert(x).hidden_states[-1]
        x = x.view(x.shape[0], -1)
        punc_logits = self.punc_fc(self.dropout(self.punc_bn(x)))
        case_logits = self.case_fc(self.dropout(self.case_bn(x)))
        return punc_logits, case_logits
    
    def _create_data_loader(self, sentences):
        sentences = [
            clean(sent, list(self.punc_dict.keys()), True) for sent in sentences
        ]
        X = preprocess_data(
            sentences,
            self.tokenizer,
            self.hparams["segment_size"]
        )
        min_required_length = self.hparams["segment_size"] // 2
        if len(X) < min_required_length:
            raise ValueError("Input has very short context!" + 
                "The input sentences must be at least {min_required_length}" +
                "token long!")
        data_loader = create_data_loader(X, False, 64)
        return data_loader

    def predict(self, sentences):
        """
        Predicts the labels for the given data.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
            A data loader object for the test data (usually).
        
        Returns
        -------
        out_tokens: list(str)
            A list of tokens found in the dataloader.
        out_preds: list(list(int))
            A list of two items; the first is the predicted labels for the 
            re-punctuation task while the other is for the re-capitalization
            task.
        out_labels: list(int)
            A list of two items; the first is the true labels for the 
            re-punctuation task while the other is for the re-capitalization
            task.
        """
        # freeze model
        self.eval()
        subwords, punc_pred, case_pred = [], [], []
        data_loader = self._create_data_loader(sentences)
        for input_batch in tqdm(data_loader, total=len(data_loader)):
            with torch.no_grad():
                input_batch = input_batch[0]
                subword_ids = input_batch[:, (self.hparams["segment_size"]-1)//2 - 1].flatten()
                subwords += self.tokenizer.convert_ids_to_tokens(subword_ids)
                if self.device == torch.device('cuda'):
                    input_batch = input_batch.cuda()
                punc_outputs, case_outputs = self.forward(input_batch)
                punc_pred += list(punc_outputs.argmax(dim=1).cpu().data.numpy().flatten())
                case_pred += list(case_outputs.argmax(dim=1).cpu().data.numpy().flatten())
        assert len(subwords) == len(punc_pred) == len(case_pred)
        
        # now, we have predictions for sub-words. Let's get the token predictions
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
        out_preds = [punc_preds, case_preds]
        return out_tokens, out_preds



if __name__ == "__main__":
    from transformers import BertTokenizer, BertModel

    # load mBERT from huggingface's transformers package
    BERT_name = "bert-base-multilingual-cased"
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_name)
    bert_model = BertModel.from_pretrained(BERT_name)

    # load trained checkpoint
    checkpoint_path = os.path.join("models", "mbert_base_cased")
    bert_punc_cap = BertPuncCap(bert_model, bert_tokenizer, checkpoint_path)

    data_test = load_file('data/mTEDx/fr/test.fr')
    x_test, y_pred_test, y_true_test = bert_punc_cap.predict(data_test)
