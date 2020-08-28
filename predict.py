# coding:utf-8
import codecs
import json
import torch
from keras.preprocessing import sequence
from bert_distill import RNN

student_model_path = "model/student.pth"
max_len = 50
vocab_path = "data/char.json"
vocab_dict = json.load(codecs.open(vocab_path, "r", "utf-8"))

# load model
model = torch.load(student_model_path)
model = model.to("cpu")
model.eval()

# predict
text = u"入住了海悦湾酒店，感觉很不错，我们住的是豪华蜜月房，房间布置的很温馨、浪漫，让我们很是开心和惊喜！房间很新很干净。"
text = [vocab_dict.get(w, 0) for w in text]

text = sequence.pad_sequences(text, max_len)
text = torch.tensor(text, dtype=torch.long).to("cpu")

with torch.no_grad():
    results = model(text)
print(results[0])
