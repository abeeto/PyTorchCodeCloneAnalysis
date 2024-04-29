import torch
from torch import nn

# input_size – The number of expected features in the input x
# hidden_size – The number of features in the hidden state h
# num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
# bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
# batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: False
# dropout – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
# bidirectional – If True, becomes a bidirectional LSTM. Default: False
# proj_size – If > 0, will use LSTM with projections of corresponding size. Default: 0

test_lstm = nn.LSTM(10, 20, 2)#(x, h, layer)
input =  torch.randn(5, 3, 10)#(size, batch_size, )
C_0 = torch.randn(2, 3, 20)
H_0 = torch.randn(2, 3, 20)


output, (hn, cn) = test_lstm(input, (H_0, C_0))
print(output, output.size())
print(hn, hn.size())
print(cn, cn.size())