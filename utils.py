# -*- coding: utf-8 -*-
import json
import codecs


def sent2id(text, vocab_dict):
    return [vocab_dict.get(w, 0) for w in text]


def load_data(data_dir=None, vocab_path=None):
    vocab_dict = json.load(codecs.open(vocab_path, "r", "utf-8"))

    x_train, y_train, text_train, x_dev, y_dev, text_dev = [], [], [], [], [], []
    with codecs.open(data_dir + "/train.txt", "r", "utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            x_train.append(sent2id(line[1], vocab_dict))
            y_train.append(int(line[0]))
            text_train.append(line[1])

    with codecs.open(data_dir + "/dev.txt", "r", "utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            x_dev.append(sent2id(line[1], vocab_dict))
            y_dev.append(int(line[0]))
            text_dev.append(line[1])

    return (x_train, y_train, text_train), (x_dev, y_dev, text_dev), max(vocab_dict.values())+1

