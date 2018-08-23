# FastBiDAF
A pytorch implemention of BiDAF based on CNN<br>
The CNN module is GLDR mostly based on [FAST READING COMPREHENSION WITH CONVNETS](https://arxiv.org/pdf/1711.04352v1.pdf)<br>
The model encoder block is based on [QAnet](https://arxiv.org/pdf/1804.09541.pdf)<br>

## Difference from the paper
1. This reposity is a combination of QAnet and GLDR. I did this because QAnet need more memory with multihead attention.<br>
2. I use beam search in the evaluate process with a beam size 5 instead of traveling all the probabilities.<br>

## Usage
1. Run ```download.sh``` to download the SQuAD dataset and GLOVE word embeddings.<br>
2. Run ```python config.py --mode preprocess``` to preprocess the data and start the first time training process.<br>
3. Run ```python config.py --mode train``` to train the model or ```python config.py --mode train --finetune True``` to finetune a model(note you should manually change the model name in main.py)<br>
4. Run ```python config.py --mode dev``` to evaluate the model and the answer file will be stored. Because this process is same as the test, I didn't duplicate the test() function.<br>


## Performance
1. The model runs fast and will have a good result after 3 hours.(TiTan XP 12GB memory)<br>
2. The best score I test is F1 73 on the dev set without any finetuning. The most hyperparameters are referred from other models, I don't know whether it's good enough.<br>

## Contributions
1. Welcome to test my code and report your performance. If you have enough time, finetuing the model is a good choice to get better result.<br>

