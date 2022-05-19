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


How this model works can be summarized in the following steps:

- BertPuncCap takes an input sentence that consists of `segment_size=32`
(by default) tokens. If the input is shorter than `segment_size`, then we are
going to pad it with the edges (both ends of the input sentence). The
`segment_size` is a hyper-parameter that you can tune.
- Then, the pre-trained BERT language model will return the representations for
the input tokens. The shape of the output should be `segment_size x model_dim`.
If you are using BERT-base, then the `model_dim=765`.
- These representations will be sent to the two linear layers for
classification. One layer should classify the punctuation after each token
while the other should classify the case.
- The loss function will be the weighted sum of the punctuation classification
loss $\text{punc-loss}$ and the capitalization classification loss 
$\text{cap-loss}$ according to the following formula where $\alpha$ is a 
hyper-parameter that you can set in your `config.yaml` file:
 
![formula](https://render.githubusercontent.com/render/math?math=loss = \alpha * \text{punc-loss} + (1 - \alpha) * \text{cap-loss})


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

> ## Working Example
>
> You can check this [notebook](https://colab.research.google.com/drive/13BHlf9ZSN9bfF_gHckq8ur9U5jiTn0gz?usp=sharing) for the different ways for
which you can use this model; also for how to get the confusion matrix of
different classes.

## Prerequisites
To install the dependencies, run the following command:
```
pip install -r requirements.txt
```
## Pre-trained Models

You can download the pre-trained models from the following table:

<div align="center">
<table>
    <thead>
        <tr>
            <th>Name</th>
            <th>Pre-trained BertPuncCap</th>
            <th>Training Data</th>
            <th>Pre-trained BERT</th>
            <th>Supported Languages</th>
        </tr>
    </thead>
    <tr>
        <td><strong>mbert_base_cased_fr</strong></td>
        <td>(
            <a href="https://drive.google.com/file/d/15f7tKvEq4BLsAjPZpkh2QXId7KAS9ZKn/view?usp=sharing"> Model</a>, 
            <a href="https://drive.google.com/file/d/1J7D02HQwZTOouaC1lx8ehVtlQpbCGYHw/view?usp=sharing"> Configuration</a>
        )</td>
        <td><a href="https://www.openslr.org/100">mTEDx</a></td>
        <td>bert-base-multilingual-cased</td>
        <td>French (fr)</td>
    </tr>
    <tr>
        <td><strong>mbert_base_cased_8langs</strong></td>
        <td>(
            <a href="https://drive.google.com/file/d/12WFBFswOfzdvW4pXSFtS9TAOPyTmZiGa/view?usp=sharing"> Model</a>, 
            <a href="https://drive.google.com/file/d/1zB_etELwrgzSl-oZiN34607xpdhGohp1/view?usp=sharing"> Configuration</a>
        )</td>
        <td><a href="https://www.openslr.org/100">mTEDx</a></td>
        <td>bert-base-multilingual-cased</td>
        <td>
            <ul>
                <li> Arabic (ar)</li>
                <li> German (de)</li>
                <li> Greek (el)</li>
                <li> French (fr)</li>
                <li> Italian (it)</li>
                <li> Spanish (es)</li>
                <li> Portuguese (pt)</li>
                <li> Russian (ru)</li>
            </ul>
        </td>
    </tr>
</table>
</div>

Now, it's very easy to use these pre-trained models; here is an example:

```python
>>> from transformers import BertTokenizer, BertModel
>>> from model import BertPuncCap
>>> 
>>> # load pre-trained mBERT from HuggingFace's transformers package
>>> BERT_name = "bert-base-multilingual-cased"
>>> bert_tokenizer = BertTokenizer.from_pretrained(BERT_name)
>>> bert_model = BertModel.from_pretrained(BERT_name)
>>> 
>>> # load trained checkpoint
>>> checkpoint_path = os.path.join("models", "mbert_base_cased")
>>> bert_punc_cap = BertPuncCap(bert_model, bert_tokenizer, checkpoint_path)
```

Now that we have loaded the model, let's use it:
```python
>>> x = ["bonsoir",
...      "notre planète est recouverte à 70 % d'océan et pourtant étrangement on a choisi de l'appeler « la Terre »"
... ]
>>> # start predicting
>>> bert_punc_cap.predict(x)
[
    'Bonsoir ,',
    "Notre planète est recouverte à 70 % d ' océan . et pourtant étrangement , on a choisi de l ' appeler « La Terre »"
]
```

## Train

To train the model, you need to use the `train.py` script. Here is how you can
do so:

``` powershell
python train.py --seed 1234 \
                --pretrained_bert bert-base-multilingual-cased \
                --optimizer Adam \
                --criterion cross_entropy \
                --alpha 0.5 \
                --dataset mTEDx \
                --langs fr \
                --save_path ./models/mbert_base_cased \
                --batch_size 1024 \
                --segment_size 32 \
                --dropout 0.3 \
                --lr 0.00001 \
                --max_epochs 50 \
                --num_validations 1 \
                --patience 1 \
                --stop_metric overall_f1
```

### Training Parameters

The following is a full list of all training parameters that can be used with
this model:

<div align="center">
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Possible Values</th>
            <th>Default</th>
        </tr>
    </thead>
    <tr>
        <td><strong>seed</strong></td>
        <td>Random seed</td>
        <td>Any positive integer value</td>
        <td>1234</td>
    </tr>
    <tr>
        <td><strong>pretrained_bert</strong></td>
        <td>The name of the pre-trained BERT model from huggingface</td>
        <td>
            <ul>
                <li>bert-base-cased</li>
                <li>bert-base-uncased</li>
                <li>bert-base-multilingual-cased</li>
                <li>bert-base-multilingual-uncased</li>
                <li>camembert-base</li>
                <li>flaubert/flaubert_base_cased</li>
            </ul>
        </td>
        <td>bert-base-multilingual-cased</td>
    </tr>
    <tr>
        <td><strong>optimizer</strong></td>
        <td>The optimizer name to train this model on</td>
        <td>Adam</td>
        <td>-</td>
    </tr>
    <tr>
        <td><strong>lr</strong></td>
        <td>The learning rate used by the optimizer.</td>
        <td>Any positive number.</td>
        <td>0.00001</td>
    </tr>
    <tr>
        <td><strong>criterion</strong></td>
        <td>The criterion used to train the model</td>
        <td>cross_entropy</td>
        <td>-</td>
    </tr>
    <tr>
        <td><strong>alpha</strong></td>
        <td>The tuning parameter of punc_loss & cap_loss</td>
        <td>any value that belongs to [0,1]</td>
        <td>0.5</td>
    </tr>
    <tr>
        <td><strong>dataset</strong></td>
        <td>The dataset used for training</td>
        <td><a href="https://www.openslr.org/100">mTEDx</a></td>
        <td>mTEDx</td>
    </tr>
    <tr>
        <td><strong>langs</strong></td>
        <td>List of languages from the dataset that you need to train your model on.</td>
        <td>Depends on the dataset</td>
        <td>fr</td>
    </tr>
    <tr>
        <td><strong>save_path</strong></td>
        <td>The relative/absolute path to save the model.</td>
        <td>A working path</td>
        <td>1234-</td>
    </tr>
    <tr>
        <td><strong>batch_size</strong></td>
        <td>The batch size for training, validating, and testing.</td>
        <td>Any positive integer value</td>
        <td>256</td>
    </tr>
    <tr>
        <td><strong>segment_size</strong></td>
        <td>The segment size of the model.</td>
        <td>Any positive integer value</td>
        <td>32</td>
    </tr>
    <tr>
        <td><strong>dropout</strong></td>
        <td>The dropout rate of the linear layers buit on top of BERT.</td>
        <td>Any value between 0 and 1.</td>
        <td>0.3</td>
    </tr>
    <tr>
        <td><strong>max_epochs</strong></td>
        <td>The maximum number of epochs to train the model.</td>
        <td>Any positive integer value</td>
        <td>50</td>
    </tr>
    <tr>
        <td><strong>num_validations</strong></td>
        <td>The number of validations to perform per epoch.</td>
        <td>Any positive integer value</td>
        <td>1</td>
    </tr>
    <tr>
        <td><strong>patience</strong></td>
        <td>The number of validations to wait for performance improvement before early stopping.</td>
        <td>Any positive integer value</td>
        <td>10</td>
    </tr>
    <tr>
        <td><strong>stop_metric</strong></td>
        <td>The name of the metric to watch for monitor to measure peformance for early stopping</td>
        <td>
            <ul>
                <li>valid_loss</li>
                <li>punc_valid_loss</li>
                <li>case_valid_loss</li>
                <li>punc_overall_f1</li>
                <li>case_overall_f1</li>
                <li>overall_f1</li>
            </ul>
        </td>
        <td>overall_f1</td>
    </tr>
</table>
</div>

### Punctuations & Cases

The list of punctuations & cases handled by this model can be seen down below:

- Punctuations:
    - COMMA
    - PERIOD
    - QUESTION
    - EXCLAMATION
    - COLON
    - SEMICOLON
    - O

- Cases:
    - F (First_Cap): When the first letter is capital.
    - A (All_Cap): When the whole token is capitalized.
    - O: Other

TO BE CONTINUED...