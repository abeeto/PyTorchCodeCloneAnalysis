import argparse
from pathlib import Path
import sys
import traceback
import json
import time
from types import ModuleType
import torch
import torchvision.models as models
import torch2ort
from ortmodel import ORTModel


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--result_dir', default=Path('result'), type=Path)
parser.add_argument('--models', default=[], nargs='+')
args = parser.parse_args()

device = 'cuda' if args.cuda else 'cpu'
n = args.num_samples
result_dir = args.result_dir

if args.models:
    names = args.models
else:
    names = [_ for _ in dir(models) if not _.startswith('_') and not isinstance(getattr(models, _), ModuleType)]


def test(name):
    result = {}

    # pytorch loading
    tic = time.time()
    model = getattr(models, name)()
    model.eval()
    model.to(device)
    t = time.time() - tic
    result['pytorch loading'] = t

    x_torch_cpu = torch.empty(1, 3, 224, 224).uniform_()
    x_numpy = x_torch_cpu.numpy()
    x_torch = x_torch_cpu.to(device)

    # export
    result_dir.mkdir(parents=True, exist_ok=True)
    result_onnx_file = str(result_dir / '{}.onnx'.format(name))
    tic = time.time()
    torch2ort.export(model, x_numpy,
                     result_onnx_file,
                     input_names=['data'],
                     output_names=['result'],
                     dynamic_axes={'data': [0, 2, 3]},  # batch size, height and width are specified to  have variable size
                     verbose=True)
    t = time.time() - tic
    result['export to onnx'] = t

    # creating ORT session
    tic = time.time()
    ort_model = ORTModel(result_onnx_file)
    t = time.time() - tic
    result['onnxruntime loading'] = t

    # PyTorch inference
    tic = time.time()
    with torch.no_grad():
        for i in range(n):
            y = model(x_torch.to(device))
    t = time.time() - tic
    result['pytorch inference'] = t

    # ORT Inference
    tic = time.time()
    for i in range(n):
        y = ort_model(x_numpy)
    t = time.time() - tic
    result['onnxruntime inference'] = t

    result['inference speedup'] = result['pytorch inference'] / result['onnxruntime inference']

    return result


results = {}
failed = []
print('---------------------------------------')
for name in names:
    print(name)
    try:
        ret = test(name)
        for k, v in ret.items():
            print(k, round(v, 5), sep='\t')
        results[name] = ret
    except KeyboardInterrupt:
        exit(0)
    except BaseException:
        print(traceback.format_exc(), file=sys.stderr)
        failed.append(name)
    print('---------------------------------------')

print('summary:')
for name, d in results.items():
    print('========================================')
    print(name)
    for k, v in d.items():
        print('', k, round(v, 5), sep='\t')
print('========================================')
print('following models are failed to benchmark:', ', '.join(failed))

json.dump(results, (result_dir / 'results.json').open('w'))
