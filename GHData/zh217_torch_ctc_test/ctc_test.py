import sys
import argparse

import torch
import torch.nn


def load_data(data_path, max_n, beta):
    test_data = torch.load(data_path)
    inp = test_data['inp']
    inp_len = torch.tensor((inp.size(0),) * inp.size(1), dtype=torch.int32)
    tar = test_data['tar']
    tar_len = test_data['tar_len']
    if max_n:
        inp[:, :, max_n - 1] = inp[:, :, max_n:].logsumexp(-1)
        inp = inp[:, :, :max_n]
        tar = tar.clamp(max=max_n - 1)

    if beta:
        inp = inp / beta

    inp = inp.log_softmax(-1)

    print(f'loaded {data_path}')
    print(f'inp.shape: {list(inp.shape)}')
    print(f'tar.shape: {list(tar.shape)}')
    print(f'inp_len: {inp_len.tolist()}')
    print(f'tar_len: {tar_len.tolist()}')
    print(f'max(inp) - min(inp): {inp.max() - inp.min()}')
    return {'inp': inp,
            'tar': tar,
            'inp_len': inp_len,
            'tar_len': tar_len}


def run_test(ctc_type, reduction, inp, tar, inp_len, tar_len):
    if ctc_type == 'cudnn' or ctc_type == 'cudnn_det':
        inp = inp.cuda().detach()
        inp_len = inp_len.cuda()
    elif ctc_type == 'plain_cuda':
        inp = inp.double().cuda().detach()
        inp_len = inp_len.long().cuda()
        tar = tar.long().cuda()
        tar_len = tar_len.long().cuda()
    else:
        inp = inp.double().detach()

    inp.requires_grad = True

    if ctc_type == 'cudnn_det':
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.deterministic = False

    loss_fn = torch.nn.CTCLoss(reduction='none')

    loss = loss_fn(inp, tar, inp_len, tar_len)

    if reduction == 'sum':
        loss.sum().backward()
    elif reduction == 'min':
        loss.min().backward()

    grad_sum = inp.grad.sum()
    grad_abs_sum = inp.grad.abs().sum()
    print(f'{ctc_type:11} '
          f'loss:  {loss[0].item():.10f}, {loss[1].item():.10f}  '
          f'grad_sum: {grad_sum.item():.10f}  '
          f'grad_abs_sum: {grad_abs_sum.item():.10f}')


parser = argparse.ArgumentParser(description='Test pytorch\'s CTC implementations')
parser.add_argument('data')
parser.add_argument('--n-class', type=int)
parser.add_argument('--beta', type=float)
parser.add_argument('--backward-reduction', default='sum')

if __name__ == '__main__':
    args = parser.parse_args()

    print('python version:', sys.version)
    print('torch version:', torch.__version__)
    print('GPU:', torch.cuda.get_device_name())

    data = load_data(args.data, args.n_class, args.beta)

    print()

    for t in ['cudnn', 'cudnn_det', 'plain_cuda', 'cpu']:
        run_test(t, args.backward_reduction, **data)
