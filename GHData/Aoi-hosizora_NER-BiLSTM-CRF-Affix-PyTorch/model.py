import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

import utils

SOS_TAG = '<SOS>'
EOS_TAG = '<EOS>'


def cap_feature(s: str) -> int:
    if s.lower() == s:
        return 0  # low caps
    if s.upper() == s:
        return 1  # all caps
    if s[0].upper() == s[0]:
        return 2  # first letter caps
    return 3  # one capital (not first letter)


class BiLSTM_CRF(nn.Module):
    """
    Bi-LSTM-CRF module with Word-Character-Capital-Affix feature.
    """
    def __init__(self,
        vocab_size, tag_to_id, pretrained_embedding, word_embedding_dim,  # word-level
        char_count, char_embedding_dim=50,  # character-level
        cap_feature_count=None, cap_embedding_dim=None,  # capital-feature
        prefix_counts=None, suffix_counts=None, prefix_embedding_dims=None, suffix_embedding_dims=None,  # affix-feature
        char_lstm_hidden_size=25, output_lstm_hidden_size=200,
        dropout_p=0.5, device='cuda', use_crf=True, add_cap_feature=True, add_affix_feature=True,
    ):
        super(BiLSTM_CRF, self).__init__()

        print()
        print("=" * 40)
        for k, v in {
            "vocab_size": vocab_size,
            "char_count": char_count,
            "cap_feature_count": cap_feature_count,
            "prefix_counts": prefix_counts,
            "suffix_counts": suffix_counts,
            "tag_count": len(tag_to_id),
            "pretrained_embedding_size": "[{}, {}]".format(len(pretrained_embedding), len(pretrained_embedding[0])),
            "word_embedding_dim": word_embedding_dim,
            "char_embedding_dim": char_embedding_dim,
            "cap_embedding_dim": cap_embedding_dim,
            "prefix_embedding_dims": prefix_embedding_dims,
            "suffix_embedding_dims": suffix_embedding_dims,
            "char_lstm_hidden_size": char_lstm_hidden_size,
            "output_lstm_hidden_size": output_lstm_hidden_size,
            "dropout_p": dropout_p,
            "device": device,
            "use_crf": use_crf,
            "add_cap_feature": add_cap_feature,
            "add_affix_feature": add_affix_feature,
        }.items():
            print("{:26s}: {}".format(k, v))
        print("=" * 40)
        print()

        assert add_cap_feature == False or (cap_feature_count is not None and cap_embedding_dim is not None), 'parameters about cap feature must be set'
        assert add_affix_feature == False or (prefix_counts is not None and suffix_counts is not None and prefix_embedding_dims is not None and suffix_embedding_dims is not None), 'parameters about affix feature must be set'

        self.tag_to_id = tag_to_id
        self.tagset_size = len(tag_to_id)
        self.char_lstm_hidden_size = char_lstm_hidden_size
        self.output_lstm_hidden_size = output_lstm_hidden_size
        self.device = device
        self.use_crf = use_crf
        self.add_cap_feature = add_cap_feature
        self.add_affix_feature = add_affix_feature

        # char
        self.char_embedding = nn.Embedding(char_count, char_embedding_dim)
        utils.init_embedding(self.char_embedding)
        self.char_lstm = nn.LSTM(char_embedding_dim, char_lstm_hidden_size, num_layers=1, bidirectional=True)
        utils.init_lstm(self.char_lstm)

        # cap
        if self.add_cap_feature:
            self.cap_embedding = nn.Embedding(cap_feature_count, cap_embedding_dim)
            utils.init_embedding(self.cap_embedding)

        # affix
        if self.add_affix_feature:
            self.prefix_2_embedding = nn.Embedding(prefix_counts[0], prefix_embedding_dims[0])
            self.prefix_3_embedding = nn.Embedding(prefix_counts[1], prefix_embedding_dims[1])
            self.prefix_4_embedding = nn.Embedding(prefix_counts[2], prefix_embedding_dims[2])
            self.suffix_2_embedding = nn.Embedding(suffix_counts[0], suffix_embedding_dims[0])
            self.suffix_3_embedding = nn.Embedding(suffix_counts[1], suffix_embedding_dims[1])
            self.suffix_4_embedding = nn.Embedding(suffix_counts[2], suffix_embedding_dims[2])
            utils.init_embedding(self.prefix_2_embedding)
            utils.init_embedding(self.prefix_3_embedding)
            utils.init_embedding(self.prefix_4_embedding)
            utils.init_embedding(self.suffix_2_embedding)
            utils.init_embedding(self.suffix_3_embedding)
            utils.init_embedding(self.suffix_4_embedding)
            affix_embedding_dims = sum(prefix_embedding_dims) + sum(suffix_embedding_dims)

        # word
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim)
        utils.init_embedding(self.word_embedding)
        self.word_embedding.weight = nn.Parameter(torch.FloatTensor(pretrained_embedding))  # Glove

        # output
        self.dropout = nn.Dropout(dropout_p)
        output_lstm_input_size = word_embedding_dim + char_lstm_hidden_size * 2
        if self.add_cap_feature:
            output_lstm_input_size += cap_embedding_dim
        if self.add_affix_feature:
            output_lstm_input_size += affix_embedding_dims
        # print("output_lstm_input_size:", output_lstm_input_size)
        self.output_lstm = nn.LSTM(output_lstm_input_size, output_lstm_hidden_size, num_layers=1, bidirectional=True)
        utils.init_lstm(self.output_lstm)
        self.output_linear = nn.Linear(output_lstm_hidden_size * 2, self.tagset_size)
        utils.init_linear(self.output_linear)
        if self.use_crf:
            self.transitions = nn.Parameter(torch.zeros(self.tagset_size, self.tagset_size))
            self.transitions.data[tag_to_id[SOS_TAG], :] = -10000
            self.transitions.data[:, tag_to_id[EOS_TAG]] = -10000


    def forward(self, words_in, chars_mask, chars_length, chars_d, caps, words_prefixes, words_suffixes):
        # print("words_in", words_in)             22
        # print("chars_mask", chars_mask)         22x10
        # print("chars_length", chars_length)     22 List[int]
        # print("chars_d", chars_d)               22 Dict[int, int]
        # print("caps", caps)                     22 (0-3)
        # print("words_prefixes", words_prefixes) 22x4
        # print("words_suffixes", words_suffixes) 22x4

        # char
        chars_embedding = self.char_embedding(chars_mask).transpose(0, 1)  # 10x22x50
        chars_packed = rnn_utils.pack_padded_sequence(chars_embedding, chars_length)
        chars_lstm_out_packed, _ = self.char_lstm(chars_packed)
        chars_lstm_out, chars_lstm_out_lengths = rnn_utils.pad_packed_sequence(chars_lstm_out_packed)
        chars_lstm_out = chars_lstm_out.transpose(0, 1)  # 22x10x50
        chars_embedding_tmp = torch.FloatTensor(torch.zeros((chars_lstm_out.size(0), chars_lstm_out.size(2)))).to(self.device)  # 22x50
        for i, index in enumerate(chars_lstm_out_lengths):
            chars_embedding_tmp[i] = torch.cat((chars_lstm_out[i, index-1, :self.char_lstm_hidden_size], chars_lstm_out[i, 0, self.char_lstm_hidden_size:]))
        chars_embedding = chars_embedding_tmp.clone()  # 22x50
        for i in range(chars_embedding.size(0)):
            chars_embedding[chars_d[i]] = chars_embedding_tmp[i]

        # cap
        if self.add_cap_feature:
            cap_embedding = self.cap_embedding(caps)  # 22x10

        # affix
        if self.add_affix_feature:
            words_prefix_2, words_prefix_3, words_prefix_4 = words_prefixes[:, 1], words_prefixes[:, 2], words_prefixes[:, 3]  # 22
            words_suffix_2, words_suffix_3, words_suffix_4 = words_suffixes[:, 1], words_suffixes[:, 2], words_suffixes[:, 3]
            affix_embeddings = [self.prefix_2_embedding(words_prefix_2), self.prefix_3_embedding(words_prefix_3), self.prefix_4_embedding(words_prefix_4),
                                self.suffix_2_embedding(words_suffix_2), self.suffix_3_embedding(words_suffix_3), self.suffix_4_embedding(words_suffix_4)]  # 22x16

        # word
        words_embedding = self.word_embedding(words_in)  # 22x100

        # output
        embedding = torch.cat((words_embedding, chars_embedding), 1)
        if self.add_cap_feature:
            embedding = torch.cat((embedding, cap_embedding), 1)
        if self.add_affix_feature:
            embedding = torch.cat((embedding, affix_embeddings[0], affix_embeddings[1], affix_embeddings[2], affix_embeddings[3], affix_embeddings[4], affix_embeddings[5]), 1)
        embedding = embedding.unsqueeze(1)  # 22x1x256
        embedding = self.dropout(embedding)
        lstm_out, _ = self.output_lstm(embedding)  # 22x1x400
        lstm_out = lstm_out.view(len(words_in), -1)  # 22x400
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.output_linear(lstm_out)  # 22x19 (feature count)
        return lstm_feats


    def _crf_forward_alg(self, feats):
        # calculate in log domain
        # feats is len(sentence) * tagset_size
        # initialize alpha with a Tensor with values all equal to -10000.
        def log_sum_exp(vec):
            # vec 2D: 1 * tagset_size
            max_score = vec[0, utils.argmax(vec)]
            max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
            return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_alphas[0][self.tag_to_id[SOS_TAG]] = 0.
        forward_var = Variable(init_alphas).to(self.device)
        for feat in feats:
            emit_score = feat.view(-1, 1)
            tag_var = forward_var + self.transitions + emit_score
            max_tag_var, _ = torch.max(tag_var, dim=1)
            tag_var = tag_var - max_tag_var.view(-1, 1)
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1)
        terminal_var = (forward_var + self.transitions[self.tag_to_id[EOS_TAG]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        return alpha


    def _crf_score_sentence(self, feats, tags):
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * tagset_size
        r = torch.LongTensor(range(feats.size()[0])).to(self.device)
        pad_SOS_TAGs = torch.cat([torch.LongTensor([self.tag_to_id[SOS_TAG]]).to(self.device), tags])
        pad_EOS_TAGs = torch.cat([tags, torch.LongTensor([self.tag_to_id[EOS_TAG]]).to(self.device)])
        score = torch.sum(self.transitions[pad_EOS_TAGs, pad_SOS_TAGs]) + torch.sum(feats[r, tags])
        return score


    def _viterbi_decode(self, feats):
        backpointers = []
        # analogous to forward
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_alphas[0][self.tag_to_id[SOS_TAG]] = 0
        forward_var = Variable(init_alphas).to(self.device)
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            viterbivars_t = Variable(torch.FloatTensor(viterbivars_t)).to(self.device)
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_id[EOS_TAG]]
        terminal_var.data[self.tag_to_id[EOS_TAG]] = -10000.
        terminal_var.data[self.tag_to_id[SOS_TAG]] = -10000.
        best_tag_id = utils.argmax(terminal_var.unsqueeze(0))
        path_score = terminal_var[best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_id[SOS_TAG]
        best_path.reverse()
        return path_score, best_path


    def calc_loss(self, feats, gt_tags):
        # print(feats)   22x19
        # print(gt_tags) 22
        if self.use_crf:
            forward_score = self._crf_forward_alg(feats)  # FloatTensor
            gold_score = self._crf_score_sentence(feats, gt_tags)  # FloatTensor
            neg_log_likelihood = forward_score - gold_score
        else:
            neg_log_likelihood = nn.functional.cross_entropy(feats, gt_tags)
        return neg_log_likelihood


    def decode_targets(self, feats):
        if self.use_crf:
            score, tag_seq = self._viterbi_decode(feats)
        else:
            score, tag_seq = torch.max(feats, 1)
            tag_seq = [i.item() for i in list(tag_seq.cpu().data)]
        return score, tag_seq
