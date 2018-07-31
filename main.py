from config import config, device
from preproc import preproc
from absl import app
import math
import os
import numpy as np
import ujson as json
import re
from collections import Counter
import string
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.cuda
from model import FastBiDAF,EMA
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(message)s')


class SQuADDataset:
    def __init__(self, data_file,train=True):
        with open(data_file, "r") as fh:
            self.data = json.load(fh)
        self.data_size = len(self.data)
        self.indices = list(range(self.data_size))
        self.train=train

    def gen_batches(self, batch_size, shuffle=True, pad_id=0):
        if shuffle:
            np.random.shuffle(self.indices)
        for batch_start in np.arange(0, self.data_size - self.data_size % batch_size, batch_size):
            batch_indices = self.indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(batch_indices, pad_id)

    def _one_mini_batch(self, indices, pad_id):
        context_word = self.dynamic_padding('context_tokens', indices, pad_id)
        question_word = self.dynamic_padding('ques_tokens', indices, pad_id)
        context_char = self.dynamic_padding('context_chars', indices, pad_id, ischar=True)
        question_char = self.dynamic_padding('ques_chars', indices, pad_id, ischar=True)
        y1s = [self.data[i]['y1s'] for i in indices]
        y2s = [self.data[i]['y2s'] for i in indices]
        ids = [self.data[i]['id'] for i in indices]

        res = (torch.Tensor(context_word).long(), torch.Tensor(context_char).long(), torch.Tensor(question_word).long(),
               torch.Tensor(question_char).long(), torch.Tensor(y1s).long(), torch.Tensor(y2s).long(),
               torch.Tensor(ids).long())
        if self.train:
            return res
        else:
            lengths = [len(self.data[i]['context_tokens']) for i in indices]
            return res,lengths

    def dynamic_padding(self, key_word, indices, pad_id, ischar=False):
        max_len = 0
        sample = []
        for i in indices:
            sample.append(self.data[i][key_word])
            max_len = max(max_len, len(self.data[i][key_word]))
        if ischar:
            pads = [pad_id] * config.word_len
            pad_sample = [ids + [pads] * (max_len - len(ids)) for ids in sample]
        else:
            pad_sample = [ids + [pad_id] * (max_len - len(ids)) for ids in sample]
        return pad_sample

    def __len__(self):
        return self.data_size


def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score,
                                            prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_batch(model, loss_func, eval_file, dataset, it_num,is_eval=False):
    answer_dict = {}
    metrics={}
    losses = 0
    for batch,lengths in tqdm(dataset, total=it_num):
        (contex_word, contex_char, question_word, question_char, y1, y2, ids) = batch
        contex_word, contex_char, question_word, question_char = contex_word.to(device), contex_char.to(
            device), question_word.to(device), question_char.to(device)
        p1, p2 = model(contex_word, contex_char, question_word, question_char)
        # y1, y2 = y1.to(device), y2.to(device)
        # loss1 = loss_func(p1, y1)
        # loss2 = loss_func(p2, y2)
        # loss = loss1 + loss2
        # losses+=loss.item()
        if is_eval:
            p1, p2 = beam_search(p1, p2,lengths)
            answer_dict_, _ = convert_tokens(
                eval_file, ids.tolist(), p1, p2)
            answer_dict.update(answer_dict_)
    loss = losses/it_num
    if is_eval:
        metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    return metrics


# def beam_search(p1s, p2s,lengths):
#     a1 = []
#     a2 = []
#     for i in range(p1s.shape[0]):
#         p1 = p1s[i]
#         p2 = p2s[i]
#         indice1, indice2 = -1, -1
#         max = -1
#         for i1 in range(lengths[i]):
#             for i2 in range(i1,lengths[i]):
#                 if p1[i1] * p2[i2] > max:
#                     max = p1[i1] * p2[i2]
#                     indice1, indice2 = i1, i2
#         a1.append(indice1)
#         a2.append(indice2)
#     return a1, a2

def max_k(p, length,beam_size):
    beam_size=min(length,beam_size)
    max_k_indices = [-1] * beam_size
    max_k_values = [-1] * beam_size
    for i in range(length):
        for j in range(beam_size):
            if p[i] > max_k_values[j]:
                if j == beam_size - 1:
                    max_k_values[j] = p[i]
                    max_k_indices[j] = i
                elif j==0:
                    max_k_values = [p[i]] + max_k_values[j:-1]
                    max_k_indices = [i] + max_k_indices[j:-1]
                else:
                    max_k_values = max_k_values[:j] + [p[i]] + max_k_values[j:-1]
                    max_k_indices = max_k_indices[:j] + [i] + max_k_indices[j:-1]
                break

    return max_k_indices

def beam_search(p1s, p2s,lengths,beam_size=5):
    a1 = []
    a2 = []

    for i in range(p1s.shape[0]):
        p1 = p1s[i]
        p2 = p2s[i]
        max_p1=max_k(p1,lengths[i],beam_size)
        max_p2=max_k(p2,lengths[i],beam_size)
        max=-1
        m1,m2=-1,-1
        for index1 in max_p1:
            for index2 in max_p2:
                if index1<=index2 and p1[index1]*p2[index2]>max:
                    max=p1[index1]*p2[index2]
                    m1=index1
                    m2=index2
        a1.append(m1)
        a2.append(m2)

    return a1, a2


def train(config):
    logger = logging.getLogger("QAnet")
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char2idx_file, "r") as fh:
        char_dict = json.load(fh)
    # with open(config.train_eval_file, "r") as fh:
    #     train_eval_file = json.load(fh)
    # with open(config.dev_eval_file, "r") as fh:
    #     dev_eval_file = json.load(fh)
    # with open(config.dev_meta, "r") as fh:
    #     meta = json.load(fh)
    train_log = open(config.train_log, "w")

    # dev_total = meta["total"]
    logger.info("Building model...")

    train_dataset = SQuADDataset(config.train_record_file)
    train_it_num = len(train_dataset) // config.batch_size
    # dev_dataset = SQuADDataset(config.dev_record_file,train=False)
    # dev_it_num=len(dev_dataset)//config.val_batch_size
    # dev_dataset=dev_dataset.gen_batches(config.val_batch_size,shuffle=False)


    lr = config.learning_rate

    char_vocab_size = len(char_dict)
    del char_dict
    model = FastBiDAF(config.char_dim, char_vocab_size, config.word_len, config.glove_dim, word_mat,
                      config.emb_dim, config.kernel_size,config.encoder_block_num,config.model_block_num,0.3).to(device)

    if config.finetune:
        model.load_state_dict(torch.load(os.path.join(config.save_dir, "model_2.53.pkl")))

    model.train()
    parameters = filter(lambda param: param.requires_grad, model.parameters())


    optimizer = optim.Adam(betas=(0.8, 0.999), eps=1e-7, weight_decay=3e-6, params=parameters,lr=0.001)
    # ema=EMA(config.decay)
    # for name,parameter in model.named_parameters():
    #     if parameter.requires_grad:
    #         ema.register(name, parameter.data)

    loss_func=torch.nn.CrossEntropyLoss()

    steps = 0
    patience = 0
    losses=0
    min_loss=10000
    # optimizer=optim.Adam(params=parameters,lr=0.001)
    for epoch in range(config.epochs):
        batches = train_dataset.gen_batches(config.batch_size,shuffle=True)
        for batch in tqdm(batches, total=train_it_num):
            optimizer.zero_grad()
            (contex_word, contex_char, question_word, question_char, y1, y2, ids) = batch
            contex_word, contex_char, question_word, question_char = contex_word.to(device), contex_char.to(
                device), question_word.to(device), question_char.to(device)
            p1, p2 = model(contex_word, contex_char, question_word, question_char)
            y1, y2 = y1.to(device), y2.to(device)
            loss1 = loss_func(p1, y1)
            loss2 = loss_func(p2, y2)
            loss = loss1 + loss2
            losses+=loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,config.grad_clip)
            optimizer.step()
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         param.data = ema(name, param.data)

            if (steps + 1) % config.checkpoint == 0:

                losses=losses/config.checkpoint
                log_ = 'itration {} train loss {}\n'.format(steps, losses)
                logger.info(log_)
                train_log.write(log_)
                # del contex_word, contex_char, question_word, question_char, y1, y2, ids
                # metric = evaluate_batch(model,loss_func, dev_eval_file, dev_dataset,dev_it_num)
                # log_ = 'itration {} dev loss {}\n'.format(steps, metric['loss'])
                # logger.info(log_)
                # train_log.write(log_)
                train_log.flush()

                if losses<min_loss:
                    patience=0
                    min_loss=losses
                    fn = os.path.join(config.save_dir, "model_{}.pkl".format(min_loss))
                    torch.save(model.state_dict(), fn)
                else:
                    patience+=1
                    if patience>config.early_stop:
                        print('early stop because of val loss is continuing incresing!')
                        exit()
                losses=0

            steps += 1

    fn = os.path.join(config.save_dir, "model_final.pkl")
    torch.save(model.state_dict(), fn)


def test(config):
    pass


def dev(config):
    logger = logging.getLogger("QAnet")
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char2idx_file, "r") as fh:
        char_dict = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)

    dev_log = open(config.dev_log, "w")
    logger.info("Building model...")
    dev_dataset = SQuADDataset(config.dev_record_file,train=False)
    dev_it_num = len(dev_dataset) // config.val_batch_size
    dev_dataset = dev_dataset.gen_batches(config.val_batch_size, shuffle=False)
    char_vocab_size = len(char_dict)
    del char_dict
    model = FastBiDAF(config.char_dim, char_vocab_size, config.word_len, config.glove_dim, word_mat,
                      config.emb_dim, config.kernel_size,config.encoder_block_num,config.model_block_num).to(device)

    model.load_state_dict(torch.load(os.path.join(config.save_dir, "model_2.08.pkl")))
    model.eval()
    loss_func = torch.nn.NLLLoss()
    metric = evaluate_batch(model, loss_func, dev_eval_file, dev_dataset, dev_it_num,is_eval=True)
    log_ = "dev_loss {:8f} F1 {:8f} EM {:8f}\n".format(metric["loss"], metric["f1"],
                                                       metric["exact_match"])
    logger.info(log_)
    dev_log.write(log_)
    # print(metric['loss'])
    dev_log.flush()


def main(_):
    if config.mode == "train":
        train(config)
    elif config.mode == "preprocess":
        preproc(config)
    elif config.mode == "debug":
        config.epochs = 20
        config.batch_size = 5
        config.val_batch_size = 1
        config.checkpoint = 10
        config.period = 1
        train(config)
    elif config.mode == "test":
        test(config)
    elif config.mode == "dev":
        dev(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == '__main__':
    app.run(main)

