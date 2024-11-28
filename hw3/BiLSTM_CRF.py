import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size):
        super(BiLSTM, self).__init__()
        # initialize word embeddings
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # bidirectional LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        # linear layer to map LSTM output to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        # embed the input sentence
        embeds = self.word_embeds(sentence)
        # pass through LSTM
        lstm_out, _ = self.lstm(embeds)
        # map LSTM output to tag space
        out = self.hidden2tag(lstm_out)
        return out


class CRF(nn.Module):
    def __init__(self, tagset_size, start_tag, stop_tag):
        super(CRF, self).__init__()
        # number of tags
        self.tagset_size = tagset_size
        # start and stop tags
        self.start_tag = start_tag
        self.stop_tag = stop_tag
        # transition matrix
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size))
        # any tag cannot transition to the start tag
        self.transitions.data[:, start_tag] = -10000
        # the stop tag cannot transition to any tag
        self.transitions.data[stop_tag, :] = -10000

    def _forward_alg(self, feats):
        """
        Compute the log sum of all possible paths(log(e^s1 + e^s2 + e^s3 + ... + e^sN)) using dynamic programming.

        :param feats: [batch_size, seq_len, tagset_size] feature matrix from BiLSTM output
        :return:
        """
        batch_size, seq_len, _ = feats.size()
        # init previous scores
        previous = torch.full((batch_size, self.tagset_size), -10000.).to(feats.device)
        # init start tag score, let start tag score be max
        previous[:, self.start_tag] = 0

        # for each token in the sequence
        for t in range(0, seq_len):
            # expand current t step emission col into a matrix
            emit_score_matrix = feats[:, t].unsqueeze(2).expand(-1, -1, self.tagset_size).transpose(1, 2)
            # expand current t step transition col into a matrix
            trans_score_matrix = self.transitions.unsqueeze(0).expand(batch_size, -1, -1)
            # expand current t step previous scores into a matrix
            expanded_previous_matrix = previous.unsqueeze(2).expand(-1, -1, self.tagset_size)
            # compute current t step scores
            scores = expanded_previous_matrix + trans_score_matrix + emit_score_matrix
            # update previous scores
            previous = torch.logsumexp(scores, dim=1)

        # add transition score from last tag to stop tag
        terminal = previous + self.transitions[:, self.stop_tag].view(1, -1).expand(batch_size, -1)
        total_score = torch.logsumexp(terminal, dim=1)
        return total_score

    def _score_sentence(self, feats, tags):
        """
        Compute the score of a given path(used for true path computing).

        :param feats: [batch_size, seq_len, tagset_size] feature matrix from BiLSTM output
        :param tags: tag sequence [batch_size, seq_len]
        :return:
        """
        batch_size, seq_len = feats.size(0), feats.size(1)
        score = torch.zeros(batch_size).to(feats.device)

        # add the emission scores
        for t in range(seq_len):
            score += feats[:, t].squeeze(1).gather(1, tags[:, t].unsqueeze(1)).squeeze(1)

        # add transition score from start tag to first tag
        first_tags = tags[:, 0]
        score += self.transitions[self.start_tag, first_tags]

        # add transition scores for each step in the sequence
        for i in range(seq_len - 1):
            current_tags = tags[:, i]
            next_tags = tags[:, i + 1]
            trans_score = self.transitions[current_tags, next_tags]
            score += trans_score

        # add transition score from last tag to stop tag
        last_tags = tags[:, -1]
        score += self.transitions[last_tags, self.stop_tag]

        return score

    def _viterbi_decode(self, feats):
        """
        Decode the best path using Viterbi algorithm.

        :param feats: [batch_size, seq_len, tagset_size] feature matrix from BiLSTM output
        :return:
        """

        ''' 实现的是单个序列非batch加载 '''
        seq_len, _ = feats.size()
        # init
        viterbi_previous = torch.full((1, self.tagset_size), -10000.).to(feats.device)
        # init start tag score, let start tag score be max
        viterbi_previous[0][self.start_tag] = 0
        backpointers = []

        for t in range(0, seq_len):
            # backpointers for this step
            bptrs_t = []
            viterbi_previous_t = []
            # 没有实现batch加载，仅一个向量，按照向量的每一维度计算（等价于一次计算整个向量）
            for tag in range(self.tagset_size):
                # emis_score(a tag dim of emission score vector)
                emis_score = feats[t, tag].view(-1, 1)
                # trans_score(a tag dim of transition score vector)
                trans_score = self.transitions[:, tag].view(1, -1)
                # scores = previous + trans_score + emis_score
                scores = viterbi_previous + trans_score + emis_score
                # viterbi算法和普通前向算法的不同之处，取each step scores的最大值
                best_tag_id = torch.argmax(scores, dim=1)
                # get scores of best_tag_id
                viterbi_previous_t.append(scores.gather(dim=1, index=best_tag_id.view(-1, 1)).squeeze(1))
                bptrs_t.append(best_tag_id)
            backpointers.append(bptrs_t)
            viterbi_previous = torch.stack(viterbi_previous_t, dim=1)

        # add transition score from last tag to stop tag
        terminal_var = viterbi_previous + self.transitions[:, self.stop_tag].view(1, -1)
        best_tag_id = torch.argmax(terminal_var, dim=1)
        best_path_scores = terminal_var.gather(dim=1, index=best_tag_id.view(-1, 1)).squeeze(1)

        # backtrack best path
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # delete start tag
        best_path.pop()
        # reverse so that the first element is the first tag of the sequence
        best_path.reverse()
        best_paths = torch.stack(best_path, dim=0)

        return best_path_scores, best_paths

    def neg_log_likelihood(self, feats, tags):
        """
        Compute negative log likelihood loss.

        :param feats: [batch_size, seq_len, tagset_size] feature matrix from BiLSTM output
        :param tags: Tag sequence [batch_size, seq_len]
        :return:
        """
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.mean(forward_score - gold_score)  # mean for mini-batch input

    def forward(self, feats):
        """
        Decode the best path.

        :param feats: [batch_size, seq_len, tagset_size] feature matrix from BiLSTM output
        :return:
        """
        scores, paths = self._viterbi_decode(feats)
        return scores, paths


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, start_tag, stop_tag):
        super(BiLSTM_CRF, self).__init__()
        self.bi_rnn = BiLSTM(vocab_size, embedding_dim, hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, start_tag, stop_tag)

    def forward(self, sentence):
        rnn_out = self.bi_rnn(sentence)
        scores, tag_seq = self.crf(rnn_out)
        return scores, tag_seq

    def loss(self, sentence, tags):
        """
        Compute the loss for BiLSTM-CRF model.

        :param sentence: input sentence indices [batch_size, seq_len]
        :param tags: ground truth tag indices [batch_size, seq_len]
        :return:
        """
        rnn_out = self.bi_rnn(sentence)
        total_loss = self.crf.neg_log_likelihood(rnn_out, tags)
        return total_loss
