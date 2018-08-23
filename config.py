import os
import absl.flags as flags
import torch
from main import train,dev
from preproc import preproc
from absl import app

'''
The content of this file is mostly copied from https://github.com/HKUST-KnowComp/R-Net/blob/master/config.py
'''

home = os.path.expanduser(".")
target_dir = 'data'
train_file = os.path.join(home, target_dir, "squad", "train-v1.1.json")
dev_file = os.path.join(home, target_dir, "squad", "dev-v1.1.json")
test_file = os.path.join(home, target_dir, "squad", "dev-v1.1.json")
glove_word_file = os.path.join(home, target_dir, "glove", "glove.840B.300d.txt")

event_dir = "log"
save_dir = "model"
answer_dir = "log"
train_token_file = os.path.join(target_dir, "train.json")
dev_token_file = os.path.join(target_dir, "dev.json")
test_token_file = os.path.join(target_dir, "test.json")
word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
dev_meta = os.path.join(target_dir, "dev_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")
word2idx_file = os.path.join(target_dir, "word2idx.json")
char2idx_file = os.path.join(target_dir, "char2idx.json")
answer_file = os.path.join(answer_dir, "answer.json")

train_logi_file = os.path.join(target_dir, "train_logi.json")
dev_logi_file = os.path.join(target_dir, "dev_logi.json")
test_logi_file = os.path.join(target_dir, "test_logi.json")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(event_dir):
    os.makedirs(event_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(answer_dir):
    os.makedirs(answer_dir)

flags.DEFINE_string("mode", "dev", "preprocess/train/debug/dev/test")
flags.DEFINE_string("model", '', "model name")

flags.DEFINE_string("target_dir", target_dir, "")
flags.DEFINE_string("event_dir", event_dir, "")
flags.DEFINE_string("save_dir", save_dir, "")
flags.DEFINE_string("train_file", train_file, "")
flags.DEFINE_string("dev_file", dev_file, "")
flags.DEFINE_string("test_file", test_file, "")
flags.DEFINE_string("glove_word_file", glove_word_file, "")

flags.DEFINE_string("train_token_file", train_token_file, "")
flags.DEFINE_string("dev_token_file", dev_token_file, "")
flags.DEFINE_string("test_token_file", test_token_file, "")
flags.DEFINE_string("word_emb_file", word_emb_file, "")
flags.DEFINE_string("char_emb_file", char_emb_file, "")
flags.DEFINE_string("train_eval_file", train_eval, "")
flags.DEFINE_string("dev_eval_file", dev_eval, "")
flags.DEFINE_string("test_eval_file", test_eval, "")
flags.DEFINE_string("dev_meta", dev_meta, "")
flags.DEFINE_string("test_meta", test_meta, "")
flags.DEFINE_string("word2idx_file", word2idx_file, "")
flags.DEFINE_string("char2idx_file", char2idx_file, "")
flags.DEFINE_string("answer_file", answer_file, "")


flags.DEFINE_integer("glove_char_size", 94, "Corpus size for Glove")
flags.DEFINE_integer("glove_word_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("char_dim", 64, "Embedding dimension for char")

flags.DEFINE_integer("word_len", 16, "Limit length for one word")
flags.DEFINE_integer("emb_dim", 100, "Dimension of connectors of each layer")
flags.DEFINE_integer("head_num", 6, "Number of heads in multi-head attention")
flags.DEFINE_integer("attention_map_dim", 64, "MultiHeadAttentnion middle dimension")
flags.DEFINE_integer("kernel_size", 3, "kernel size of CNN")
flags.DEFINE_integer("encoder_block_num", 5, "Number of blocks in the model encoder layer")
flags.DEFINE_integer("model_block_num", 17, "Number of blocks in the model encoder layer")


flags.DEFINE_integer("para_limit", 400, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
flags.DEFINE_integer("ans_limit", 30, "Limit length for answers")
flags.DEFINE_integer("test_para_limit", 1000, "Limit length for paragraph in test file")
flags.DEFINE_integer("test_ques_limit", 100, "Limit length for question in test file")
flags.DEFINE_integer("word_count_limit", -1, "Min count for word")
flags.DEFINE_integer("char_count_limit", -1, "Min count for char")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")
flags.DEFINE_boolean("is_bucket", False, "build bucket batch iterator or not")
flags.DEFINE_list("bucket_range", [40, 401, 40], "the range of bucket")

flags.DEFINE_integer("batch_size", 40, "Batch size")
flags.DEFINE_integer("epochs", 10, "Number of epochs")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_float("decay", 0.9999, "Exponential moving average decay")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_batch_size", 20, "Number of batches to evaluate the model")
flags.DEFINE_float("dropout", 0.2, "Dropout prob across the layers")
flags.DEFINE_float("dropout_char", 0.05, "Dropout prob across the layers")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("L2_norm", 3e-5, "L2 norm scale")
# flags.DEFINE_integer("hidden", 96, "Hidden size")
flags.DEFINE_integer("early_stop", 10, "Checkpoints for early stop")


flags.DEFINE_string("train_log", "log/train.log", "Log for each checkpoint")
flags.DEFINE_string("dev_log", "log/dev.log", "Log for validation")


# Extensions (Uncomment corresponding line in download.sh to download the required data)
glove_char_file = os.path.join(home, target_dir, "glove", "glove.840B.300d-char.txt")
flags.DEFINE_string("glove_char_file", glove_char_file, "Glove character embedding")
flags.DEFINE_boolean("pretrained_char", False, "Whether to use pretrained char embedding")

fasttext_file = os.path.join(home, target_dir, "fasttext", "wiki-news-300d-1M.vec")
flags.DEFINE_string("fasttext_file", fasttext_file, "Fasttext word embedding")
flags.DEFINE_boolean("fasttext", False, "Whether to use fasttext")

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
config = flags.FLAGS

def run(_):
    if config.mode == "train":
        train(config,device)
    elif config.mode == "preprocess":
        preproc(config)
        train(config,device)


    elif config.mode == "debug":
        config.epochs = 1
        config.batch_size = 5
        config.val_batch_size = 20
        config.checkpoint = 1
        config.period = 1
        train(config,device)
    elif config.mode == "test":
        pass
    elif config.mode == "dev":
        dev(config,device)
    else:
        print("Unknown mode")
        exit(0)

if __name__ == '__main__':
    app.run(run)






