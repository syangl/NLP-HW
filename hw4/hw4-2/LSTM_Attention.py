# coding=utf-8
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk import word_tokenize
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm

if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.exists('out'):
    os.mkdir('out')


def get_tokenization(sentence):
    e1 = re.findall(r'<e1>(.*)</e1>', sentence)[0]
    e2 = re.findall(r'<e2>(.*)</e2>', sentence)[0]
    sentence = sentence.replace('<e1>' + e1 + '</e1>', ' <e1> ' + e1 + ' </e1> ', 1)
    sentence = sentence.replace('<e2>' + e2 + '</e2>', ' <e2> ' + e2 + ' </e2> ', 1)
    sentence = word_tokenize(sentence)
    sentence = ' '.join(sentence)
    sentence = sentence.replace('< e1 >', '<e1>')
    sentence = sentence.replace('< e2 >', '<e2>')
    sentence = sentence.replace('< /e1 >', '</e1>')
    sentence = sentence.replace('< /e2 >', '</e2>')
    sentence = sentence.split()

    assert '<e1>' in sentence
    assert '<e2>' in sentence
    assert '</e1>' in sentence
    assert '</e2>' in sentence

    return sentence



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2024
# 常量定义
FIXED_SIZE = 128  # 128
EMBEDDING_DIM = 50  # 50
RELATION_COUNT = 19
# POS_EMBEDDING_DIM = 100  # 100
BATCH_SIZE = 30  # 30

EPOCH = 300  # 300
STRIP = 10

# 自定义数据集类
class RelationDataset(Dataset):
    def __init__(self, data_file, word_to_idx, max_len=FIXED_SIZE):
        self.sentences = []
        self.e1_positions = []
        self.e2_positions = []
        self.labels = []
        self.max_len = max_len

        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i in range(0, len(lines), 4):
            sentence = lines[i].strip().split('\t')[1]
            label = lines[i + 1].strip()

            # e1_start = sentence.find('<e1>')
            # e1_end = sentence.find('</e1>')
            # e2_start = sentence.find('<e2>')
            # e2_end = sentence.find('</e2>')
            #
            # sentence_clean = sentence.replace('<e1>', '').replace('</e1>', '').replace('<e2>', '').replace('</e2>', '')
            # words = sentence_clean.split()
            words = get_tokenization(sentence)

            # e1_pos = [min(max(j - e1_start, 0), max_len) for j in range(len(words))]
            # e2_pos = [min(max(j - e2_start, 0), max_len) for j in range(len(words))]

            indices = [word_to_idx.get(word, 1) for word in words]  # 1 is the index for unknown words

            if len(indices) > max_len:
                indices = indices[:max_len]
                # e1_pos = e1_pos[:max_len]
                # e2_pos = e2_pos[:max_len]
            else:
                indices.extend([0] * (max_len - len(indices)))  # Padding with 0s
                # e1_pos.extend([max_len] * (max_len - len(e1_pos)))
                # e2_pos.extend([max_len] * (max_len - len(e2_pos)))

            self.sentences.append(torch.tensor(indices, dtype=torch.long))
            # self.e1_positions.append(torch.tensor(e1_pos, dtype=torch.long))
            # self.e2_positions.append(torch.tensor(e2_pos, dtype=torch.long))
            self.labels.append(label)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return (self.sentences[idx],
                # self.e1_positions[idx],
                # self.e2_positions[idx],
                self.labels[idx])



class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.randn(1, hidden_dim, 1))

    def forward(self, lstm_out):
        M = self.tanh(lstm_out)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1)
        # r = torch.sum(lstm_out * alpha, 1)
        r = torch.bmm(lstm_out.transpose(1, 2), alpha).squeeze(dim=-1)
        r = self.tanh(r)
        return r

# 注意力机制模块
class BiLSTM_Attention(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim): #, pos_size, pos_dim):
        super(BiLSTM_Attention, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim

        self.hidden_dim = hidden_dim
        self.tag_size = output_size

        # self.pos_size = pos_size
        # self.pos_dim = pos_dim

        self.word_embeds = nn.Embedding(self.input_size, self.embedding_dim)

        # self.pos1_embeds = nn.Embedding(self.pos_size, self.pos_dim)
        # self.pos2_embeds = nn.Embedding(self.pos_size, self.pos_dim)

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim // 2,#+ self.pos_dim * 2, hidden_size=self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, bias=True, dropout=0, batch_first=True)

        self.attention_layer = Attention(self.hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_size, bias=True)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.)

        self.dropout_emb = nn.Dropout(p=0.3)
        self.dropout_lstm = nn.Dropout(p=0.3)
        self.dropout_att = nn.Dropout(p=0.5)


    def forward(self, index): # , pos1, pos2):
        word_embeds = self.word_embeds(index)
        # pos1_embeds = self.pos1_embeds(pos1)
        # pos2_embeds = self.pos2_embeds(pos2)

        # Embedding layer
        # embed_concat = torch.cat((word_embeds, pos1_embeds, pos2_embeds), dim=-1)
        # embed_concat = self.dropout_emb(embed_concat)
        embed_concat = self.dropout_emb(word_embeds)
        # BiLSTM layer
        lstm_out, _ = self.lstm(embed_concat)
        lstm_out = self.dropout_lstm(lstm_out)
        # Attention layer
        att_output = self.attention_layer(lstm_out)
        att_output = self.dropout_att(att_output)

        out = self.fc(att_output)
        return out

# 绘制定损曲线
def plot_loss_curve(train_lossv, output_dir):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_lossv) + 1)
    plt.plot(epochs, train_lossv, 'bo-', label='Training Loss')
    plt.xlabel(f"Epochs(*{STRIP})")
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.show()


# 评估函数
def evaluate(model, data_loader, criterion, device, label_encoder, output_dir):
    model.eval()
    running_loss = 0.0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        # for index, pos1, pos2, labels in data_loader:
        #     index, pos1, pos2 = index.to(device), pos1.to(device), pos2.to(device)
        for index, labels in data_loader:
            index = index.to(device)
            labels = torch.LongTensor(label_encoder.transform(labels)).to(device)

            # outputs = model(index, pos1, pos2)
            outputs = model(index)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    avg_loss = running_loss / len(data_loader)

    # 计算宏平均F1分数
    report = classification_report(true_labels, predicted_labels, output_dict=True, zero_division=1)
    macro_f1 = report['macro avg']['f1-score']

    print(f'Test Loss: {avg_loss:.3f} | Test Macro F1: {macro_f1 * 100:.2f}%')
    with open(os.path.join(output_dir, 'result.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Test Loss: {avg_loss:.3f} | Test Macro F1: {macro_f1 * 100:.2f}%\n")
        # 保存预测结果（预测标签+真实标签的形式）
        for idx, pred, true_label in zip(range(8001, 8001 + len(predicted_labels)), predicted_labels, true_labels):
            f.write(f"{idx}\t Pred: {label_encoder.inverse_transform([pred])[0]} \t\t| True: {label_encoder.inverse_transform([true_label])[0]}\n")



# 加载数据
base_dir = "SemEval2010_task8_all_data"
train_file = os.path.join(base_dir, "SemEval2010_task8_training", "TRAIN_FILE.TXT")
test_file = os.path.join(base_dir, "SemEval2010_task8_testing_keys", "TEST_FILE_FULL.txt")

# 构建词汇表
word_counts = defaultdict(int)
with open(train_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    i = 0
    for i in range(0, len(lines), 4):
        sentence = lines[i].strip().split('\t')[1]
        # sentence_clean = sentence.replace('<e1>', '').replace('</e1>', '').replace('<e2>', '').replace('</e2>', '')
        # words = sentence_clean.split()
        words = get_tokenization(sentence)
        for word in words:
            word_counts[word] += 1

word_to_idx = {'PAD': 0, 'UNK': 1, '<e1>': 2, '</e1>': 3, '<e2>': 4, '</e2>': 5}
idx = 6
for word, count in word_counts.items():
    # if count >= 5:
    if word not in ['<e1>', '</e1>', '<e2>', '</e2>']:
        word_to_idx[word] = idx
        idx += 1

# 标签编码
label_list = [
    'Other', 'Cause-Effect(e1,e2)', 'Component-Whole(e1,e2)', 'Entity-Destination(e1,e2)',
    'Product-Producer(e1,e2)', 'Entity-Origin(e1,e2)', 'Member-Collection(e1,e2)',
    'Message-Topic(e1,e2)', 'Content-Container(e1,e2)', 'Instrument-Agency(e1,e2)',
    'Cause-Effect(e2,e1)', 'Component-Whole(e2,e1)', 'Entity-Destination(e2,e1)',
    'Product-Producer(e2,e1)', 'Entity-Origin(e2,e1)', 'Member-Collection(e2,e1)',
    'Message-Topic(e2,e1)', 'Content-Container(e2,e1)', 'Instrument-Agency(e2,e1)'
]
label_encoder = LabelEncoder()
label_encoder.fit(label_list)

# 创建数据集和数据加载器
train_dataset = RelationDataset(train_file, word_to_idx)
test_dataset = RelationDataset(test_file, word_to_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Train Set Size: {len(train_loader)*BATCH_SIZE}")

# 初始化模型
vocab_size = len(word_to_idx)
model = BiLSTM_Attention(input_size=vocab_size,
                         output_size=RELATION_COUNT,
                         embedding_dim=EMBEDDING_DIM,
                         hidden_dim=EMBEDDING_DIM * 2,
                         # pos_size=FIXED_SIZE * 2 + 1,
                         # pos_dim=POS_EMBEDDING_DIM,
                         ).to(device)
with open('model/model_structure.txt', 'w', encoding='utf-8') as f:
    f.write(str(model))

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
lambda_l2 = 1e-5  # L2正则化强度
optimizer = optim.Adadelta(model.parameters(), lr=1.0, weight_decay=lambda_l2)

# 训练和评估
train_lossv = []
output_dir = "out/"

TRAIN_FLAG = True
if TRAIN_FLAG == True:
    for epoch in tqdm(range(EPOCH)):
        model.train()
        running_loss = 0.0

        # for index, pos1, pos2, labels in train_loader:
        #     index, pos1, pos2 = index.to(device), pos1.to(device), pos2.to(device)
        for index, labels in train_loader:
            index = index.to(device)
            labels = torch.LongTensor(label_encoder.transform(labels)).to(device)

            optimizer.zero_grad()
            # outputs = model(index, pos1, pos2)
            outputs = model(index)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        if epoch % STRIP == 0:
            print(f"Epoch: {epoch + 1}/{EPOCH}\tTrain Loss: {avg_train_loss:.3f}")
            train_lossv.append(avg_train_loss)

    # 绘制定损曲线
    plot_loss_curve(train_lossv, output_dir)

    # Save
    torch.save(model.state_dict(), os.path.join(output_dir, 'BiLSTM_Attention_Model.pth'))
    print("Model Saved")
# Prediction
model.eval()
model.load_state_dict(torch.load(os.path.join(output_dir, 'BiLSTM_Attention_Model.pth')))
evaluate(model, test_loader, criterion, device, label_encoder, output_dir)

