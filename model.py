import os
import torch
from torch import nn
from tqdm import tqdm

from utils import *
from data_handler import DataHandler


class BertPuncCap(nn.Module):
    def __init__(self, BERT_model, BERT_tokenizer, model_path=''):
        """
        Initializes the model.

        Parameters
        ----------
        BERT_model: transformers.PreTrainedModel
            The BERT pre-trained model at HuggingFace's `transformers` package.
        BERT_tokenizer: transformers.PreTrainedTokenizer
            A tokenizer object from the HuggingFace's `transformers` package.
        model_path: str
            The relative/absolute path for the trained model (default: '').
        """
        super(BertPuncCap, self).__init__()
        # save important params
        self.bert = BERT_model
        self.bert.config.output_hidden_states=True
        self.tokenizer = BERT_tokenizer
        self.hparams = parse_yaml(os.path.join(model_path, "config.yaml"))
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # get needed params
        dropout_rate = self.hparams["dropout"]
        hidden_size = self.bert.config.hidden_size
        segment_size = self.hparams["segment_size"]
        punc_size = len(self.hparams["class_to_punc"])
        case_size = len(self.hparams["class_to_case"])
        # create handler for data
        self._data_handler = DataHandler(self.tokenizer, segment_size,
            self.hparams["punc_to_class"], self.hparams["case_to_class"])
        # build one extra layer
        self.punc_bn = nn.BatchNorm1d(segment_size*hidden_size)
        self.punc_fc = nn.Linear(segment_size*hidden_size, punc_size)
        self.case_bn = nn.BatchNorm1d(segment_size*hidden_size)
        self.case_fc = nn.Linear(segment_size*hidden_size, case_size)
        self.dropout = nn.Dropout(dropout_rate)
        # load trained model's stat_dict
        self.load_state_dict(load_checkpoint(model_path, self.device))

    def forward(self, x):
        """
        Returns
        -------
        out_punc: torch.Tensor
            The output of the re-punctuation task. The size is
            (batch_size, #punctuation_labels). 
        out_case: torch.Tensor
            The output of the re-capitalization task. The size is
            (batch_size, #capitalization_labels).
        """
        x = self.bert(x).hidden_states[-1]
        x = x.view(x.shape[0], -1)
        punc_logits = self.punc_fc(self.dropout(self.punc_bn(x)))
        case_logits = self.case_fc(self.dropout(self.case_bn(x)))
        return punc_logits, case_logits
   
    def _get_labels(self, sentences, batch_size=64):
        """
        Predicts the labels for the given data.

        Parameters
        ----------
        sentences: list(str)
            A list of cleaned sentences that will be labeled.
        batch_size: int
            The batch size for processing the test data (default 64).
        
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
        subwords, punc_pred, case_pred = [], [], []
        data_loader = \
            self._data_handler.create_test_dataloader(sentences, batch_size)
        # Get predictions for sub-words
        sg_size = self.hparams["segment_size"]
        for input_batch in tqdm(data_loader, total=len(data_loader)):
            with torch.no_grad():
                input_batch = input_batch[0]
                subword_ids = input_batch[:, (sg_size-1)//2 - 1].flatten()
                subwords += self.tokenizer.convert_ids_to_tokens(subword_ids)
                # move data & model to device
                self = self.to(self.device)
                input_batch  = input_batch.to(self.device)
                punc_outputs, case_outputs = self.forward(input_batch)
                # get the class that has the highest probability
                punc_pred += list(punc_outputs.argmax(dim=1).cpu().data.numpy().flatten())
                case_pred += list(case_outputs.argmax(dim=1).cpu().data.numpy().flatten())
        assert len(subwords) == len(punc_pred) == len(case_pred)
        # Convert sub-token predictions to full-token predictions
        out_tokens, punc_preds, case_preds = \
            convert_to_full_tokens(subwords, punc_pred, case_pred)
        return out_tokens, punc_preds, case_preds

    def predict(self, sentences, batch_size=64):
        """
        Punctuate & capitalize the given sentences. The punctuations & cases
        will be removed from the given sentences if 

        Parameters
        ----------
        sentences: list(str)
            A list of sentences to be labeled.
        A flattened list of the same input tokens.
        
        Returns
        -------
        out_sentences: list(str)
            A list of sentences labeled with punctuation & cases.
        
        Note
        ----
        Punctuations & capitalization are removed from the input sentences.
        """
        self.eval() # freeze the model
        # clean sentences from punctuations & cases
        #TODO: FIND A BETTER WAY TO CLEAN & PROCESS DATA.
        cleaned_tokens, _, _ = \
            self._data_handler._extract_tokens_labels(sentences, "Cleaning")
        cleaned_sentences = [
            " ".join(sent_tokens) for sent_tokens in cleaned_tokens
        ]
        # get labels
        out_tokens, punc_preds, case_preds = \
                                self._get_labels(sentences, batch_size)
        # Apply labels to input sentences
        out_sentences = apply_labels_to_input(
            [len(sent_tokens) for sent_tokens in cleaned_tokens],
            out_tokens,
            punc_preds,
            case_preds,
            self.hparams["class_to_punc"],
            self.hparams["class_to_case"]
        )
        return out_sentences


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
    print(data_test[54:55])
    out_sentences = bert_punc_cap.predict(data_test[54:55])
    print(out_sentences)
