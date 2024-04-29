import os
import logging
import argparse
from typing import OrderedDict

import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import ElectraForTokenClassification, ElectraTokenizer


# 필요한 import문 for ONNX
import torch.onnx

def load_tokenizer(args):
    return ElectraTokenizer.from_pretrained(args.model_name_or_path)

def get_args(pred_config):
    return torch.load(os.path.join(pred_config, 'training_args.bin'))

def read_input_file(input_dir):
    lines = []
    with open(input_dir, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    return lines


def convert_input_file_to_tensor_dataset(lines,
                                         args,
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    all_input_tokens = []
    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            # slot_label_mask.extend([0] + [pad_token_label_id] * (len(word_tokens) - 1))
            
            # use the real label id for all tokens of the word
            slot_label_mask.extend([0] * (len(word_tokens)))

        all_input_tokens.append(tokens) # 뭐지? 여기서는 뒤에 sep 안 붙는데 이 이후부터 계속 붙네?

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # print("=====================")
        # print(all_input_tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        # print("=====================")
        # print(all_input_tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)

        input_ids = input_ids + ([pad_token_id] * padding_length)

        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)
    
    return dataset, all_input_tokens

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# test code for compare tokenizer method (tokenize vs. encode)
def test_tokenizer(tokenizer):
    
    print("=========== comparing tokenizer's output ==============")
    
    text = "여기는 서대문구 노연로 80에 위치하고 있어요." 
    
    tokens = tokenizer.tokenize(text) # tokens
    token_ids = tokenizer.encode(text) # ids
    
    print("tokenized: ", token_ids)
    print("encoded :", tokens)

# export torchScript model
def convert_to_script(model_torch, input_torch, SCRIPT_OUTPUT_PATH):
    
    # input_torch(OrderdDict)를 이렇게 넘겨주기 위해서는 modeling_electra.py 파일이 수정되어 있어야 한다 
    model_traced_script = torch.jit.trace(model_torch, input_torch)
    print("traced compelete!")
    torch.jit.save(model_traced_script, SCRIPT_OUTPUT_PATH)
    
    print("TorchScript converted & exported complete!")

# compare script output vs. 
def compare_with_script(output_torch, input_torch, SCRIPT_OUTPUT_PATH):
    
    model_script = torch.jit.load(SCRIPT_OUTPUT_PATH)
    model_script.to(device)  
    model_script.eval()

    output_script = model_script(input_torch)

    print("=============comparing output values via torch.ne (torch vs. torchScript)=============")
    a = torch.ne(output_script[0], output_torch[0])
    x = a.nonzero() 
    if (len(x) == 0):
        print("output is compeletely same!")
        
    else: # meaning that torch model is not perfectly converted into torchScript
        print("torch.ne result: ", x)
        print("something wrong with conversion...") 

    
def convert_to_onnx(model_torch, inputs_torch, ONNX_OUTPUT_PATH):
    
    output_names = ['output']
    shape = (1,1)
    ordered_input_names = ['input_ids', 'attention_mask']
    
    dynamic_axes = {
    "input_ids": {0: 'batch', 1:'sequence'},
    "attention_mask": {0: 'batch', 1:'sequence'},
    "output": {0: 'batch'}
    }
    
    # 메모리에 연속 배열 반환 
    inputs_onnx = {
        k: np.ascontiguousarray(v.detach().cpu().numpy()) for k, v in inputs_torch.items()
    }
    
    # 모델 변환 
    torch.onnx.export(
        model_torch,                            # 실행될 모델 
        args = tuple(inputs_torch.values()),    # 모델 입력값 (튜플 또는 여러 입력값 가능)
        f = ONNX_OUTPUT_PATH,                   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
        input_names =ordered_input_names,       # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
        output_names = output_names,
        dynamic_axes=dynamic_axes,              # 가변적인 길이를 가진 차원
        do_constant_folding=True,               # 최적화시 상수폴딩을 사용할지의 여부
        # training = TrainingMode.EVAL          # 의 경우 default가 EVAL로 되어 있음 
        opset_version=10,                       # 모델을 변환할 때 사용할 ONNX 버전       
        verbose=False
    )
    
    print("ONNX converted & exported complete!")


if __name__ == "__main__":
    
    MODEL_DIR = './model' # edit configuration에서 working directory 확인하기 (파이참에서 상대 경로 쓸 경우)
    SCRIPT_OUTPUT_PATH = './KoELECTRA_script.pt'
    ONNX_OUTPUT_PATH = './KoELECTRA_onnx.onnx'

    device = get_device()

    model_torch = ElectraForTokenClassification.from_pretrained(MODEL_DIR, torchscript=True)  # Config will be automatically loaded from model_dir

    # prepare input
    args = get_args(MODEL_DIR)


    # Convert input file to TensorDataset
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    tokenizer = load_tokenizer(args)

    lines = read_input_file('./sample.txt')
    dataset, all_input_tokens = convert_input_file_to_tensor_dataset(lines, args, tokenizer, pad_token_label_id)

    '''
    dataset: (B, E)
    '''
    
    all_input_ids = dataset.tensors[0]
    all_attention_mask = dataset.tensors[1]
    
    model_torch.to(device)
    model_torch.eval()
    
    input_torch = OrderedDict()
    input_torch["input_ids"] = all_input_ids[0].unsqueeze(0) # extends dimension on passed index
    input_torch["attention_mask"] = all_input_ids[0].unsqueeze(0)
    
    with torch.no_grad():
        
        output_torch = model_torch(**input_torch)
        convert_to_onnx(model_torch, input_torch, ONNX_OUTPUT_PATH)
        # convert_to_script(model_torch, input_torch, SCRIPT_OUTPUT_PATH)
        # compare_with_script(output_torch, input_torch, SCRIPT_OUTPUT_PATH)
