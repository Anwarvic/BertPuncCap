# BertPuncCap

This is a simple PoC (proof-of-concept) model built to restore punctuation & 
capitalization from a given text. In other words, given a text with no
punctuations and no capitalization, this model is able to restore the needed
punctuations and capitalization to make the text human-readable.

BertPuncCap is PyTorch model built on top of a pre-trained Google's
[BERT](https://arxiv.org/pdf/1810.04805) model by creating two linear layers
that perform the two tasks simultaneously. One layer is responsible for the 
**re-punctuation** task and the other is responsible for the **re-punctuation**
task as shown in the following figure: 


<div align="center">
    <img src="https://i.ibb.co/B6gfKz7/Bert-Punc-Cap.png" width=500px>
</div>

The method of how BertPuncCap works can be summarized in the following steps:

- BertPuncCap takes an input that consists of `segment_size` tokens. If the
input is shorter than `segment_size`, then we are going to pad it with the
edges. The `segment_size` is a hyper-parameter that you can control.
- Then, the pre-trained BERT will return its representations for the input
tokens. The shape of the output should be `segment_size x model_dim`.
- These representations will be sent to the two linear layers for
classification. One layer should classify the punctuation after each token
while the  other should classify the case.


> **Note:**
>
> BertPuncCap was inspired by [BertPunc](https://github.com/nkrnrnk/BertPunc)
with the following differences:
> 
> - BertPunc only handles punctuation restoration task, while this model
handles both **punctuation restoration** & **re-capitalization**.
> - BertPunc only handles COMMA, PERIOD and QUESTION_MARK, while this model
handles three more punctuations EXCLAMATION, COLON, SEMICOLON. And you can 
add yours if you want. It's totally configurable. 
> - BertPunc is not compatible with HuggingFace `transformers` package, while 
this model does.
> - BertPunc doesn't provide any pre-trained model, while this model provides
many.


## Prerequisites
To install the dependencies, run the following command:
```
pip install -r requirements.txt
```


## Pre-trained Models

You can download the pre-trained models from the following table:

