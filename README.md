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
3. Run ```python config.py --mode train``` to train the model or ```python config.py --mode train --model modelname``` to finetune a model.(eg. ```python config.py --mode train --model mode_final.pkl```)<br>
4. Run ```python config.py --mode dev --model modelname``` to evaluate the model and the answer file will be stored. Because this process is same as the test, I didn't duplicate the test() function.<br>


## Performance
1. The model runs fast and will have a good result after 3 hours.(TiTan XP 12GB memory)<br>
2. The best score I test is F1 74 on the dev set without any finetuning. The most hyperparameters are referred from other models, I don't know whether it's good enough.<br>
<table>
<thead>
<tr>
<th>Model</th>
<th>EM</th>
<th>F1</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">QAnet</td>
<td>73.6</td>
<td>82.7</td>
</tr>
<tr>
<td align="center">GLDR</td>
<td>68.2</td>
<td>77.2</td>
</tr>
<tr>
<td align="center">MyModel</td>
<td>63.7</td>
<td>74.3</td>
</tr>
</tbody>
</table>

## Contributions
1. Welcome to test my code and report your performance. If you have enough time, finetuing the model(dropout, conv layer number, etc.) is a good choice to get better results.<br>

