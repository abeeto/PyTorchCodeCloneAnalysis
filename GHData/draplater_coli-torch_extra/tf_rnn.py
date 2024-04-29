from io import BytesIO
from typing import Tuple

import torch

from torch import nn, Tensor
import torch.nn.init as I
import torch.nn.functional as F
from torch.jit import script_method, ScriptModule
from torch.nn import LayerNorm, ModuleList, Sequential, LSTMCell, Parameter

from coli.torch_extra.seq_utils import sort_sequences, unsort_sequences, pad_timestamps_and_batches


class TFCompatibleLSTMCell(ScriptModule):
    __constants__ = ["forget_bias"]

    def __init__(self, input_size, hidden_size,
                 # input_keep_prob=1.0, recurrent_keep_prob=1.0,
                 weight_initializer=I.orthogonal_,
                 forget_bias=1.0,
                 activation=torch.tanh):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.input_keep_prob = float(input_keep_prob)
        # self.recurrent_keep_prob = float(recurrent_keep_prob)
        self.weight_initializer = weight_initializer
        self.activation = activation
        self.forget_bias = forget_bias

        self.linearity = nn.Linear(input_size + self.hidden_size, 4 * hidden_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight_initializer(self.linearity.weight)
        I.zeros_(self.linearity.bias)

    @script_method
    def forward(self, inputs: Tensor, h_prev: Tensor, c_prev: Tensor) -> Tuple[Tensor, Tensor]:
        lstm_matrix = self.linearity(torch.cat([inputs, h_prev], -1))
        i, j, f, o = torch.split(
            lstm_matrix,
            lstm_matrix.shape[1] // 4,
            dim=1)
        c = torch.sigmoid(f + self.forget_bias) * c_prev + \
            torch.sigmoid(i) * self.activation(j)
        h = torch.sigmoid(o) * self.activation(c)
        return h, c

    def extra_repr(self):
        return 'input_size={}, hidden_size={}'.format(
            self.input_size, self.hidden_size
        )


class NormalLSTMCell(ScriptModule):
    __constants__ = ["forget_bias"]

    def __init__(self, input_size, hidden_size,
                 # input_keep_prob=1.0, recurrent_keep_prob=1.0,
                 weight_initializer=I.orthogonal_,
                 forget_bias=1.0,
                 activation=torch.tanh):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.input_keep_prob = float(input_keep_prob)
        # self.recurrent_keep_prob = float(recurrent_keep_prob)
        self.weight_initializer = weight_initializer
        self.activation = activation
        self.forget_bias = forget_bias

        # self.wx = nn.Linear(input_size, 4 * hidden_size, bias=True)
        # self.wh = nn.Linear(self.hidden_size, 4 * hidden_size, bias=True)
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        try:
            from allennlp.nn.initializers import block_orthogonal
            block_orthogonal(self.weight_ih, [self.hidden_size, self.input_size])
            block_orthogonal(self.weight_hh, [self.hidden_size, self.hidden_size])
        except ImportError:
            self.weight_initializer(self.weight_ih)
            self.weight_initializer(self.weight_hh)
        I.zeros_(self.bias_hh)
        I.zeros_(self.bias_ih)

    @script_method
    def forward(self, inputs: Tensor, h_prev: Tensor, c_prev: Tensor) -> Tuple[Tensor, Tensor]:
        lstm_matrix = torch.matmul(inputs, self.weight_ih.t()) + self.bias_ih + \
                      torch.matmul(h_prev, self.weight_hh.t()) + self.bias_hh
        i, j, f, o = torch.split(
            lstm_matrix,
            lstm_matrix.shape[1] // 4,
            dim=-1)
        c = torch.sigmoid(f + self.forget_bias) * c_prev + \
            torch.sigmoid(i) * self.activation(j)
        h = torch.sigmoid(o) * self.activation(c)
        return h, c
        # noinspection PyProtectedMember
        # return torch.lstm_cell(
        #     inputs, [h_prev, c_prev],
        #     self.weight_ih, self.weight_hh,
        #     self.bias_ih, self.bias_hh
        # )

    def extra_repr(self):
        return 'input_size={}, hidden_size={}'.format(
            self.input_size, self.hidden_size
        )


class DynamicRNN(ScriptModule):
    __constants__ = [
        "hidden_size", "input_keep_prob", "recurrent_keep_prob", "go_forward",
        "batch_first"]

    def __init__(self, cell,
                 input_keep_prob=1.0,
                 recurrent_keep_prob=1.0,
                 go_forward=True,
                 ):
        super().__init__()
        self.cell = cell
        self.go_forward = go_forward
        self.input_keep_prob = float(input_keep_prob)
        self.recurrent_keep_prob = float(recurrent_keep_prob)
        self.hidden_size = self.cell.hidden_size

    @script_method
    def forward(self, sequence_tensor: Tensor, batch_lengths: Tensor):
        total_timesteps = int(sequence_tensor.shape[0])
        batch_size = int(sequence_tensor.shape[1])
        # output_accumulator = torch.zeros([total_timesteps, batch_size, self.hidden_size],
        #                                  dtype=sequence_tensor.dtype, device=sequence_tensor.device)
        output_accumulator = torch.zeros([0, batch_size, self.hidden_size],
                                         dtype=sequence_tensor.dtype, device=sequence_tensor.device)
        full_batch_previous_memory = torch.zeros([batch_size, self.hidden_size],
                                                 dtype=sequence_tensor.dtype, device=sequence_tensor.device)
        full_batch_previous_state = torch.zeros([batch_size, self.hidden_size],
                                                dtype=sequence_tensor.dtype, device=sequence_tensor.device)

        if self.training and self.input_keep_prob < 1.0:
            # input_noise = F.dropout(torch.ones([1, batch_size, sequence_tensor.shape[-1]],
            #                                    dtype=sequence_tensor.dtype, device=sequence_tensor.device),
            #                         1 - self.input_keep_prob, self.training
            #                         )
            # sequence_tensor = sequence_tensor * input_noise
            sequence_tensor = F.dropout(sequence_tensor, 1 - self.input_keep_prob, self.training)

        if self.training and self.recurrent_keep_prob < 1.0:
            recurrent_noise = F.dropout(torch.ones([batch_size, self.hidden_size],
                                                   dtype=sequence_tensor.dtype, device=sequence_tensor.device),
                                        1 - self.recurrent_keep_prob, self.training
                                        )
        else:
            recurrent_noise = torch.ones([batch_size, self.hidden_size],
                                         dtype=sequence_tensor.dtype, device=sequence_tensor.device)

        current_length_index = (batch_size - 1) if self.go_forward else 0
        for timestep in range(total_timesteps):
            # The index depends on which end we start.
            index = timestep if self.go_forward else total_timesteps - timestep - 1

            # What we are doing here is finding the index into the batch dimension
            # which we need to use for this timestep, because the sequences have
            # variable length, so once the index is greater than the length of this
            # particular batch sequence, we no longer need to do the computation for
            # this sequence. The key thing to recognise here is that the batch inputs
            # must be _ordered_ by length from longest (first in batch) to shortest
            # (last) so initially, we are going forwards with every sequence and as we
            # pass the index at which the shortest elements of the batch finish,
            # we stop picking them up for the computation.
            if self.go_forward:
                while current_length_index >= 0 and int(batch_lengths[current_length_index]) <= index:
                    current_length_index -= 1
            # If we're going backwards, we are _picking up_ more indices.
            else:
                # First conditional: Are we already at the maximum number of elements in the batch?
                # Second conditional: Does the next shortest sequence beyond the current batch
                # index require computation use this timestep?
                while current_length_index < (batch_size - 1) and \
                        int(batch_lengths[current_length_index + 1]) > index:
                    current_length_index += 1

            if 0 <= current_length_index < batch_size:
                timestep_output, memory = self.cell(
                    sequence_tensor[index, :(current_length_index + 1)],
                    full_batch_previous_state[:(current_length_index + 1)],
                    full_batch_previous_memory[:(current_length_index + 1)])

                if self.training and self.recurrent_keep_prob < 1.0:
                    timestep_output = timestep_output * recurrent_noise[:(current_length_index + 1)]

                # We've been doing computation with less than the full batch, so here we create a new
                # variable for the the whole batch at this timestep and insert the result for the
                # relevant elements of the batch into it.
                # full_batch_previous_memory = full_batch_previous_memory.clone()
                # full_batch_previous_state = full_batch_previous_state.clone()
                # full_batch_previous_memory[:(current_length_index + 1)] = memory
                # full_batch_previous_state[:(current_length_index + 1)] = timestep_output
                full_batch_previous_memory = torch.cat([memory[:current_length_index + 1],
                                                        full_batch_previous_memory[current_length_index + 1:]], 0)
                full_batch_previous_state = torch.cat([timestep_output[:current_length_index + 1],
                                                       full_batch_previous_state[current_length_index + 1:]], 0)
                output_accumulator = torch.cat([output_accumulator, full_batch_previous_memory.unsqueeze(0)], 0)
                # output_accumulator[index, :(current_length_index + 1)] = timestep_output

        # Mimic the pytorch API by returning state in the following shape:
        # (num_layers * num_directions, batch_size, hidden_size). As this
        # LSTM cannot be stacked, the first dimension here is just 1.
        # final_state = (full_batch_previous_state.unsqueeze(0),
        #                full_batch_previous_memory.unsqueeze(0))

        ret = output_accumulator
        return ret


class Identity(ScriptModule):
    @script_method
    def forward(self, x):
        return x


class UniDirLSTMLayer(ScriptModule):
    __constants__ = ["use_layer_norm"]

    def __init__(self,
                 cell_class,
                 input_size: int,
                 hidden_size: int,
                 input_keep_prob: float = 1.0,
                 recurrent_keep_prob: float = 1.0,
                 layer_norm=False):
        super(UniDirLSTMLayer, self).__init__()
        self.forward_layer = DynamicRNN(
            cell_class(input_size, hidden_size),
            input_keep_prob,
            recurrent_keep_prob,
            go_forward=True)
        self.use_layer_norm = layer_norm

        if layer_norm:
            self.layer_norm = LayerNorm(hidden_size)
        else:
            self.layer_norm = Identity()

    @script_method
    def forward(self, seqs, lengths):
        output_seqs = self.forward_layer(seqs, lengths)
        if self.use_layer_norm:
            output_seqs = self.layer_norm(output_seqs)
        return output_seqs


class BiLSTMLayer(ScriptModule):
    __constants__ = ["use_layer_norm", "input_keep_prob"]

    def __init__(self,
                 cell_class,
                 input_size: int,
                 hidden_size: int,
                 input_keep_prob: float = 1.0,
                 recurrent_keep_prob: float = 1.0,
                 layer_norm=False):
        super(BiLSTMLayer, self).__init__()
        self.forward_layer = DynamicRNN(
            cell_class(input_size, hidden_size),
            1.0,
            recurrent_keep_prob,
            go_forward=True)
        self.backward_layer = DynamicRNN(
            cell_class(input_size, hidden_size),
            1.0,
            recurrent_keep_prob,
            go_forward=False)
        self.use_layer_norm = layer_norm

        if layer_norm:
            self.layer_norm = LayerNorm(hidden_size * 2)
        else:
            self.layer_norm = Identity()

        self.input_keep_prob = float(input_keep_prob)

    @script_method
    def forward(self, seqs, lengths):
        if self.input_keep_prob < 1.0:
            input_noise = F.dropout(torch.ones([1, seqs.shape[1], seqs.shape[-1]],
                                               dtype=seqs.dtype, device=seqs.device),
                                    1 - self.input_keep_prob, self.training
                                    )
            seqs = seqs * input_noise
        forward_output = self.forward_layer(seqs, lengths)
        backward_output = self.backward_layer(seqs, lengths)
        output_seqs = torch.cat([forward_output, backward_output], -1)
        if self.use_layer_norm:
            output_seqs = self.layer_norm(output_seqs)
        return output_seqs


class LSTM(ScriptModule):
    """
    A standard stacked Bidirectional LSTM where the LSTM layers
    are concatenated between each layer. The only difference between
    this and a regular bidirectional LSTM is the application of
    variational dropout to the hidden states of the LSTM.
    Note that this will be slower, as it doesn't use CUDNN.
    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    hidden_size : int, required
        The dimension of the outputs of the LSTM (for each direction).
    num_layers : int, required
        The number of stacked Bidirectional LSTMs to use.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .
    """

    cell_class = NormalLSTMCell
    __constants__ = ["num_layers", "use_layer_norm", "layers"]

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 input_keep_prob: float = 1.0,
                 recurrent_keep_prob: float = 1.0,
                 layer_norm=False,
                 first_dropout=0,
                 bidirectional=True
                 ) -> None:
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_layer_norm = layer_norm
        self.bidirectional = True
        self.output_dim = self.hidden_size * (2 if self.bidirectional else 1)

        lstm_input_size = input_size
        layers = []

        for layer_index in range(num_layers):
            args = (self.cell_class, lstm_input_size, hidden_size,
                    input_keep_prob if layer_index >= 1 else (1 - first_dropout),
                    recurrent_keep_prob, layer_norm)
            if bidirectional:
                layers.append(BiLSTMLayer(*args))
                lstm_input_size = hidden_size * 2
            else:
                layers.append(UniDirLSTMLayer(*args))
                lstm_input_size = hidden_size

        self.layers = ModuleList(layers)

    @script_method
    def run_rnn(self, seqs, lengths):
        seqs = seqs.transpose(0, 1).contiguous()
        output_seqs = seqs
        for layer in self.layers:
            output_seqs = layer(output_seqs, lengths)
        return output_seqs.transpose(0, 1).contiguous()

    def forward(self,  # pylint: disable=arguments-differ
                seqs: Tensor,
                lengths: Tensor,
                is_sorted=False,
                return_all=False
                ):
        original_shape = seqs.shape
        if not is_sorted:
            seqs, lengths, unsort_idx = sort_sequences(seqs, lengths, False)
        output_seqs = self.run_rnn(seqs, lengths)
        if not is_sorted:
            # noinspection PyUnboundLocalVariable
            output_seqs = unsort_sequences(output_seqs, unsort_idx, original_shape, unpack=False)
        else:
            # noinspection PyUnboundLocalVariable
            output_seqs = pad_timestamps_and_batches(output_seqs, original_shape)
        # noinspection PyUnboundLocalVariable
        return output_seqs

    @classmethod
    def load_func(cls, data):
        f = BytesIO(data)
        map_location = "cpu"
        import inspect
        # get map_location from stack
        map_location_2 = inspect.currentframe().f_back.f_locals.get("map_location")
        if map_location_2:
            map_location = map_location_2
        ret = torch.jit.load(f, map_location=map_location)

        parameters = dict(ret.named_parameters())
        try:
            ret.output_dim = parameters["layers.1.w_2c.bias"].shape[-1] + parameters["layers.1.w_2p.bias"].shape[-1]
        except KeyError:
            ret.output_dim = parameters["layers.1.w_2.bias"].shape[-1]

        ret.__class__ = cls
        return ret

    def __reduce__(self):
        f = BytesIO()
        torch.jit.save(self, f)
        f.seek(0)
        return self.__class__.load_func, (f.read(),)


if __name__ == '__main__':
    lstm = LSTM(100, 100, 3, 0.5, 0.5, False)
    # lengths, _ = torch.sort(-torch.randint(0, 9, [8]))
    # lstm(torch.randn(8, 10, 100), -lengths)
    torch.jit.save(lstm, "/tmp/dadfa")
