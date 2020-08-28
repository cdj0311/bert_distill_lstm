# bert_distill_lstm
论文： Distilling Task-Specific Knowledge from BERT into Simple Neural Networks.

代码修改自：https://github.com/qiangsiwei/bert_distill

将bert模型在特定任务下蒸馏到简单神经网络以加快推理速度。

1. 对bert进行finetune, 生成teacher.pth模型

python bert_finetune.py

2. 训练蒸馏模型，生成student.pth模型

python bert_distill.py

3. 利用蒸馏模型进行预测

python predict.py

