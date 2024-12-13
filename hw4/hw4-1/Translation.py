import os

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import torchtext
from tqdm import tqdm

torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import spacy
import random
import math
import time
from pathlib import Path

# 设置随机种子以确保结果可重复
SEED = 20241212
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 加载分词工具
spacy_en = spacy.load('en_core_web_sm')
spacy_zh = spacy.load('zh_core_web_sm')


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tokenize_zh(text):
    return [tok.text for tok in spacy_zh.tokenizer(text)]


# 构建词汇表
SRC_TOKENIZER = get_tokenizer(tokenize_en)
TRG_TOKENIZER = get_tokenizer(tokenize_zh)


def yield_tokens(data_iter, tokenizer):
    """
    生成器函数，用于从数据迭代器中提取并分词文本
    """
    for data_sample in data_iter:
        yield tokenizer(data_sample)


def load_data(data_path):
    train_src = (line.strip() for line in open(Path(data_path) / 'train.en', encoding='utf-8'))
    train_trg = (line.strip() for line in open(Path(data_path) / 'train.zh', encoding='utf-8'))

    valid_src = (line.strip() for line in open(Path(data_path) / 'valid.en', encoding='utf-8'))
    valid_trg = (line.strip() for line in open(Path(data_path) / 'valid.zh', encoding='utf-8'))

    test_src = (line.strip() for line in open(Path(data_path) / 'test.en', encoding='utf-8'))
    test_trg = (line.strip() for line in open(Path(data_path) / 'test.zh', encoding='utf-8'))

    return ( # TODO: For debug
        {'src': list(train_src)[:5], 'trg': list(train_trg)[:5]},
        {'src': list(valid_src)[:5], 'trg': list(valid_trg)[:5]},
        {'src': list(test_src)[:5], 'trg': list(test_trg)[:5]}
    )
    # return (
    #     {'src': list(train_src), 'trg': list(train_trg)},
    #     {'src': list(valid_src), 'trg': list(valid_trg)},
    #     {'src': list(test_src), 'trg': list(test_trg)}
    # )


# 假设data_path是你的数据路径
train_data, valid_data, test_data = load_data(data_path='data/en-zh/')

SRC_VOCAB = build_vocab_from_iterator(yield_tokens(train_data['src'], SRC_TOKENIZER),
                                      specials=["<unk>", "<pad>", "<bos>", "<eos>"])
TRG_VOCAB = build_vocab_from_iterator(yield_tokens(train_data['trg'], TRG_TOKENIZER),
                                      specials=["<unk>", "<pad>", "<bos>", "<eos>"])

SRC_VOCAB.set_default_index(SRC_VOCAB["<unk>"])
TRG_VOCAB.set_default_index(TRG_VOCAB["<unk>"])

def collate_batch(batch):
    """
    将一批次的数据填充到相同的长度，并转换为tensor
    """
    src_list, trg_list = [], []
    for (_src, _trg) in batch:
        src_list.append(_src)
        trg_list.append(_trg)
    src_pad_list = pad_sequence(src_list, padding_value=SRC_VOCAB["<pad>"], batch_first=True)
    trg_pad_list = pad_sequence(trg_list, padding_value=TRG_VOCAB["<pad>"], batch_first=True)
    return src_pad_list.to(device), trg_pad_list.to(device)

class TranslationDataset(Dataset):
    def __init__(self, src_lines, trg_lines, src_vocab, trg_vocab, max_len=100):
        self.src_lines = src_lines
        self.trg_lines = trg_lines
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_line = self.src_lines[idx]
        trg_line = self.trg_lines[idx]

        # 分词
        src_tensor = torch.tensor([self.src_vocab[token] for token in SRC_TOKENIZER(src_line)[:self.max_len]],
                                  dtype=torch.long)
        trg_tensor = torch.tensor([self.trg_vocab[token] for token in TRG_TOKENIZER(trg_line)[:self.max_len]],
                                  dtype=torch.long)

        # 添加起始符和结束符
        src_tensor = torch.cat((torch.tensor([self.src_vocab["<bos>"]], dtype=torch.long), src_tensor,
                                torch.tensor([self.src_vocab["<eos>"]], dtype=torch.long)))
        trg_tensor = torch.cat((torch.tensor([self.trg_vocab["<bos>"]], dtype=torch.long), trg_tensor,
                                torch.tensor([self.trg_vocab["<eos>"]], dtype=torch.long)))

        return src_tensor, trg_tensor


train_dataset = TranslationDataset(train_data['src'], train_data['trg'], SRC_VOCAB, TRG_VOCAB)
valid_dataset = TranslationDataset(valid_data['src'], valid_data['trg'], SRC_VOCAB, TRG_VOCAB)
test_dataset = TranslationDataset(test_data['src'], test_data['trg'], SRC_VOCAB, TRG_VOCAB)

BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        :param d_model: 嵌入向量的维度
        :param dropout: dropout 概率，默认为 0.1
        :param max_len: 最大序列长度，默认为 5000
        """
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)
        # 初始化一个形状为 (max_len, d_model) 的张量，存储位置编码
        pe = torch.zeros(max_len, d_model)
        # 初始化从 0 到 max_len 的位置索引张量
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算位置编码的频率部分
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 计算位置编码的正弦部分
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算位置编码的余弦部分
        pe[:, 1::2] = torch.cos(position * div_term)
        # 调整位置编码的形状，使其适合输入
        pe = pe.unsqueeze(0)
        # 注册位置编码为缓冲区，以便在模型中使用
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将输入 x 与位置编码相加
        x = x + self.pe[:, :x.shape[1], :].expand_as(x)
        return x



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "embed_size必须被heads整除"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 把嵌入向量拆分为多个头
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        # queries shape: (N, query_len, heads, heads_dim), keys shape: (N, key_len, heads, heads_dim)->energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            # 填充为负无穷
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len), values shape: (N, value_len, heads, heads_dim) -> out: (N, query_len, heads, head_dim)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)  # 展开最后两维
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(d_model=embed_size, max_len=max_length)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # N, seq_length = x.shape
        # positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        embed_x = self.word_embedding(x)
        out = self.dropout(self.position_embedding(embed_x))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads=heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size=embed_size,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    """解码器部分，由多个解码器块堆叠而成"""

    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size,  max_len=max_length)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size=embed_size,
                             heads=heads,
                             forward_expansion=forward_expansion,
                             dropout=dropout,
                             )
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, target_mask):
        # positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        embed_x = self.word_embedding(x)
        out = self.dropout(self.position_embedding(embed_x))

        for layer in self.layers:
            out = layer(out, enc_out, enc_out, src_mask, target_mask)

        out = self.fc_out(out)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_size,
            num_layers,
            forward_expansion,
            heads,
            dropout,
            device,
            max_length,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            device=device,
            forward_expansion=forward_expansion,
            dropout=dropout,
            max_length=max_length,
        )
        self.decoder = Decoder(
            trg_vocab_size=trg_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            device=device,
            max_length=max_length,
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # 不是填充位置的1，填充位置为0
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        # 不是填充位置的1，填充位置为0
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # 下三角矩阵，用于屏蔽未来的时间步
        trg_len_mask = torch.tril(torch.ones((trg_len, trg_len))).bool().to(self.device)
        trg_mask = trg_pad_mask & trg_len_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


# 模型参数
SRC_PAD_IDX = SRC_VOCAB["<pad>"]
TRG_PAD_IDX = TRG_VOCAB["<pad>"]
EMBED_SIZE = 256
NUM_LAYERS = 3
FORWARD_EXPANSION = 4
HEADS = 8
DROPOUT = 0.10
MAX_LENGTH = 100

model = Transformer(
    src_vocab_size=len(SRC_VOCAB),
    trg_vocab_size=len(TRG_VOCAB),
    src_pad_idx=SRC_PAD_IDX,
    trg_pad_idx=TRG_PAD_IDX,
    embed_size=EMBED_SIZE,
    num_layers=NUM_LAYERS,
    forward_expansion=FORWARD_EXPANSION,
    heads=HEADS,
    dropout=DROPOUT,
    device=device,
    max_length=MAX_LENGTH,
).to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


def train_epoch(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, trg in dataloader:
        optimizer.zero_grad()

        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in dataloader:
            output = model(src, trg[:-1, :])

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def plot_loss(train_lossv, train_pplv, valid_lossv, valid_pplv, path):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)  # (nrows, ncols, index)
    plt.plot(train_lossv, label='train loss', color='blue')
    plt.plot(valid_lossv, label='valid loss', color='red')

    plt.subplot(1, 2, 2) # (nrows, ncols, index)
    plt.plot(train_pplv, label='train ppl', color='blue')
    plt.plot(valid_pplv, label='valid ppl', color='red')

    plt.savefig(path)

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')
train_lossv = []
train_pplv = []
valid_lossv = []
valid_pplv = []
for epoch in tqdm(range(N_EPOCHS)):

    start_time = time.time()

    train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_loader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer.pth')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    train_lossv.append(train_loss)
    valid_lossv.append(valid_loss)
    train_pplv.append(math.exp(train_loss))
    valid_pplv.append(math.exp(valid_loss))

# 保存图像
if not os.path.exists('fig'):
    os.mkdir('fig')
plot_loss(train_lossv, train_pplv, valid_lossv, valid_pplv, 'fig/loss.png')

# 测试模型
model.load_state_dict(torch.load('transformer.pth'))

test_loss = evaluate(model, test_loader, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')



