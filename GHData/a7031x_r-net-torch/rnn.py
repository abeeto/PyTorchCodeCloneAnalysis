import torch
import torch.nn as nn
import torch.nn.functional as F
import data
import func
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class StackedBRNN(nn.Module):
    """Stacked Bi-directional RNNs.
    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        """Encode either padded or non-padded sequences.
        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        x_mask = 1 - x_mask
        if x_mask.data.sum() == 0:
            # No padding necessary.
            output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            # Pad if we care or if its during eval.
            output = self._forward_padded(x, x_mask)
        else:
            # We don't care.
            output = self._forward_unpadded(x, x_mask)

        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise, encoding that handles
        padding.
        """
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class EncoderBase(nn.Module):
    def forward(self, src, lengths=None, encoder_state=None):
        raise NotImplementedError


class RNNEncoder(EncoderBase):
    def __init__(self, input_size, num_layers, hidden_size, bidirectional, dropout=0.0, type='lstm', use_bridge=False, batch_first=True):
        super(RNNEncoder, self).__init__()
        hidden_size = hidden_size//2 if bidirectional else hidden_size
        rnn = {
            'lstm': nn.LSTM,
            'gru': nn.GRU
        }
        self.type = type
        self.rnn = rnn[type](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first)


    def sort_batch(self, data, lengths):
        sorted_lengths, sorted_idx = lengths.sort(descending=True)
        sorted_data = data[sorted_idx]
        return sorted_data, sorted_lengths


    def forward_ordered(self, src, lengths, encoder_state):
        full_batch_size = lengths.shape[0]
        valid_batch_size = (lengths != 0).sum()
        packed_emb = pack(src[:valid_batch_size], lengths[:valid_batch_size], batch_first=self.rnn.batch_first)
        memory_bank, encoder_final = self.rnn(packed_emb, encoder_state)
        memory_bank = unpack(memory_bank, batch_first=self.rnn.batch_first)[0]
        encoder_final = tuple([func.pad_zeros(s, full_batch_size, dim=1) for s in encoder_final]) if isinstance(encoder_final, tuple) else func.pad_zeros(encoder_final, full_batch_size, dim=1)
        memory_bank = func.pad_zeros(memory_bank, full_batch_size)
        return memory_bank, encoder_final


    def forward(self, src, lengths, encoder_state=None, ordered=False):
        if lengths is not None and lengths.max() != lengths.min():
            if ordered:
                return self.forward_ordered(src, lengths, encoder_state)
            else:
                sorted_lengths, perm_idx = lengths.sort(descending=True)
                sorted_src = src[perm_idx]
                if encoder_state is not None:
                    encoder_state = encoder_state[perm_idx]
                memory_bank, encoder_final = self.forward_ordered(sorted_src, sorted_lengths, encoder_state)
                _, odx = perm_idx.sort()
                memory_bank = memory_bank[odx]
                encoder_final = tuple([s[:, odx, :] for s in encoder_final]) if isinstance(encoder_final, tuple) else encoder_final[:, odx, :]
        else:
            memory_bank, encoder_final = self.rnn(src, encoder_state)
        return memory_bank, encoder_final


if __name__ == '__main__':
    embeddings = [
        [0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3],
        [-0.1, 0.05, -0.2],
        [0.2, -0.1, 0.1],
        [-0.12, -0.2, 0.15]
    ]
    embeddings = nn.Embedding.from_pretrained(torch.tensor(embeddings))
    embeddings.padding_idx = 0
    seq0 = [1, 2, 3]
    seq1 = [4, 2, 0]
    seq2 = [3, 0, 0]
    seq3 = [0, 0, 0]
    s0 = torch.tensor([seq0, seq1, seq2, seq3])
    l0 = torch.tensor([3, 2, 1, 0])
    s1 = torch.tensor([seq3, seq1, seq2, seq0])
    l1 = torch.tensor([0, 2, 1, 3])


    for type in ['lstm', 'gru']:
        rnn = RNNEncoder(input_size=3, num_layers=3, hidden_size=4, bidirectional=True, type=type)
        m0, t0 = rnn(embeddings(s0), l0, ordered=True)
        m1, t1 = rnn(embeddings(s1), l1, ordered=False)
        d0 = m1[3] - m0[0]
        d1 = m1[1] - m0[1]
        d2 = m1[2] - m0[2]
        d3 = m1[0] - m0[3]
        assert d0.abs().sum().tolist() == 0
        assert d1.abs().sum().tolist() == 0
        assert d2.abs().sum().tolist() == 0
        assert d3.abs().sum().tolist() == 0
        if not isinstance(t0, tuple):
            t0 = (t0,)
            t1 = (t1,)
        d0 = t1[0][:,3,:] - t0[0][:,0,:]
        d1 = t1[-1][:,1] - t0[-1][:,1]
        d2 = t1[0][:,2] - t0[0][:,2,:]
        d3 = t1[-1][:,0] - t0[-1][:,3]
        assert d0.abs().sum().tolist() == 0
        assert d1.abs().sum().tolist() == 0
        assert d2.abs().sum() == 0
        assert d3.abs().sum() == 0