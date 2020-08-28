# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from keras.preprocessing import sequence
import pickle
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer
from utils import load_data
from bert_finetune import BertClassification


USE_CUDA = torch.cuda.is_available()
if USE_CUDA: torch.cuda.set_device(0)
FTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
device = torch.device('cuda' if USE_CUDA else 'cpu')

class RNN(nn.Module):
    def __init__(self, x_dim, e_dim, h_dim, o_dim):
        super(RNN, self).__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(0.2)
        self.emb = nn.Embedding(x_dim, e_dim, padding_idx=0)
        self.lstm = nn.LSTM(e_dim, h_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(h_dim * 2, o_dim)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        embed = self.dropout(self.emb(x))
        out, _ = self.lstm(embed)
        hidden = self.fc(out[:, -1, :])
        return self.softmax(hidden), self.log_softmax(hidden)


class Teacher(object):
    def __init__(self, bert_model='bert-base-chinese', max_seq=128, model_dir=None):
        self.max_seq = max_seq
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.model = torch.load(model_dir)
        self.model.eval()

    def predict(self, text):
        tokens = self.tokenizer.tokenize(text)[:self.max_seq]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_seq - len(input_ids))
        input_ids = torch.tensor([input_ids + padding], dtype=torch.long).to(device)
        input_mask = torch.tensor([input_mask + padding], dtype=torch.long).to(device)
        logits = self.model(input_ids, input_mask, None)
        return F.softmax(logits, dim=1).detach().cpu().numpy()


def train_student(bert_model_dir="/data0/sina_up/dajun1/src/doc_dssm/sentence_bert/bert_pytorch",
                  teacher_model_path="./model/teacher.pth",
                  student_model_path="./model/student.pth",
                  data_dir="data/hotel",
                  vocab_path="data/char.json",
                  max_len=50,
                  batch_size=64,
                  lr=0.002,
                  epochs=10,
                  alpha=0.5):

    teacher = Teacher(bert_model=bert_model_dir, model_dir=teacher_model_path)
    teach_on_dev = True
    (x_tr, y_tr, t_tr), (x_de, y_de, t_de), vocab_size = load_data(data_dir, vocab_path)

    l_tr = list(map(lambda x: min(len(x), max_len), x_tr))
    l_de = list(map(lambda x: min(len(x), max_len), x_de))

    x_tr = sequence.pad_sequences(x_tr, maxlen=max_len)
    x_de = sequence.pad_sequences(x_de, maxlen=max_len)

    with torch.no_grad():
        t_tr = np.vstack([teacher.predict(text) for text in t_tr])
        t_de = np.vstack([teacher.predict(text) for text in t_de])

    with open(data_dir+'/t_tr', 'wb') as fout: pickle.dump(t_tr,fout)
    with open(data_dir+'/t_de', 'wb') as fout: pickle.dump(t_de,fout)

    model = RNN(vocab_size, 256, 256, 2)

    if USE_CUDA: model = model.cuda()
    opt = optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.NLLLoss()
    mse_loss = nn.MSELoss()
    for epoch in range(epochs):
        losses, accuracy = [], []
        model.train()
        for i in range(0, len(x_tr), batch_size):
            model.zero_grad()
            bx = Variable(LTensor(x_tr[i:i + batch_size]))
            by = Variable(LTensor(y_tr[i:i + batch_size]))
            bl = Variable(LTensor(l_tr[i:i + batch_size]))
            bt = Variable(FTensor(t_tr[i:i + batch_size]))
            py1, py2 = model(bx)
            loss = alpha * ce_loss(py2, by) + (1-alpha) * mse_loss(py1, bt)  # in paper, only mse is used
            loss.backward()
            opt.step()
            losses.append(loss.item())
        for i in range(0, len(x_de), batch_size):
            model.zero_grad()
            bx = Variable(LTensor(x_de[i:i + batch_size]))
            bl = Variable(LTensor(l_de[i:i + batch_size]))
            bt = Variable(FTensor(t_de[i:i + batch_size]))
            py1, py2 = model(bx)
            loss = mse_loss(py1, bt)
            if teach_on_dev:
                loss.backward()             
                opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            for i in range(0, len(x_de), batch_size):
                bx = Variable(LTensor(x_de[i:i + batch_size]))
                by = Variable(LTensor(y_de[i:i + batch_size]))
                bl = Variable(LTensor(l_de[i:i + batch_size]))
                _, py = torch.max(model(bx, bl)[1], 1)
                accuracy.append((py == by).float().mean().item())
        print(np.mean(losses), np.mean(accuracy))
    torch.save(model, student_model_path)


if __name__ == "__main__":
    train_student() 

