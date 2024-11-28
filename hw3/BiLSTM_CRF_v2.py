import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF


class BiLSTM_CRF_v2(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(BiLSTM_CRF_v2, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)

    def get_mask(self, lengths):
        mask = []
        max_length = max(lengths)
        for length in lengths:
            mask.append([1 for i in range(length)] + [0 for j in range(max_length - length)])
        return torch.tensor(mask, dtype=torch.bool)

    def forward(self, sentences, lengths, tags):
        embeds = self.word_embeds(sentences)
        # packed_sentences = pack_padded_sequence(embeds, lengths=lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(embeds)
        # out, _ = pad_packed_sequence(out, batch_first=True, total_length=self.max_length)
        feats = self.hidden2tag(out)
        mask = self.get_mask(lengths)
        scores = self.crf(feats, tags, mask.to(sentences.device))
        loss = (-1) * scores
        return loss

    def predict(self, sentences, lengths):
        embeds = self.word_embeds(sentences)
        # packed_sentences = pack_padded_sequence(embeds, lengths=lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(embeds)
        # out, _ = pad_packed_sequence(out, batch_first=True, total_length=self.max_length)
        feats = self.hidden2tag(out)
        mask = self.get_mask(lengths).to(sentences.device)
        return self.crf.decode(feats, mask)