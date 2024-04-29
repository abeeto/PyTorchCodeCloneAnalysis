#!/usr/bin/env python
# coding: utf-8

import os
import time
from unittest.mock import patch

import torch
from torch import package
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc", use_fast=False)

def run_model(model_, s0, s1):
    token_ids_mask = tokenizer.encode_plus(s0, s1, return_tensors="pt")

    token_ids_mask = {k:v.to(model.device) for k,v in token_ids_mask.items()}

    st = time.time()
    classification_logits = model_(**token_ids_mask)
    print(f"Execution time: {1000*(time.time() - st):.2f} ms")

    paraphrase_results = torch.softmax(classification_logits[0], dim=1).cpu().tolist()[0]

    return f"{round(paraphrase_results[1] * 100)}% paraphrase"


model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

if torch.cuda.is_available():
  model = model.to('cuda')

print('Original Model')
print(run_model(model, sequence_0, sequence_1))
print(run_model(model, sequence_0, sequence_2))

## Export with TorchPackage

package_path = 'model.pt'

venv_dir = 'venv'

to_mock = [f'transformers.models.{s}.**' for s in os.listdir(f'./{venv_dir}/lib/python3.8/site-packages/transformers/models') if not (s.startswith('_') or s in ['bert', 'auto'])]

to_mock += [
  'transformers.pipelines.**',
  'tokenizers.tokenizers.**',
  'transformers.trainer*.**',
  'transformers.integrations.**',
  'transformers.optimization.**',
  'transformers.data.**',
  ]

with package.PackageExporter(package_path) as bert_package_exp:
    bert_package_exp.intern([
      'transformers.**',
      'packaging.**',
      'importlib_metadata.**',
      'tokenizers.**',
      'handler.**',
      ], exclude=to_mock )
    bert_package_exp.extern([
      'torch.**',
      'sys',
      'io',
      '__future__.**',
      '_queue',
    ])
    bert_package_exp.mock([
      'torchaudio.**',
      'huggingface_hub',
      'PIL.**',
      'yaml.**',
      'numpy.**',
      'urllib3.**',
      'requests.**',
      'pkg_resources.**',
      'regex.**',
      'six.**',
      'sacremoses.**',
      'absl.**',
      'idna.**',
      'tqdm.**',
      'filelock.**',
      'google.**',
      'IPython.display.**',
      'certifi.**',
      'charset_normalizer.**',
      ] + to_mock)

    bert_package_exp.save_pickle("model", "model.pkl", model)

