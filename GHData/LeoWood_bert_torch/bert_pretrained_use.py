# coding=utf-8

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForNextSentencePrediction
import numpy as np

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

model_mask_en = BertForMaskedLM.from_pretrained('bert-base-uncased')
model_mask_cn = BertForMaskedLM.from_pretrained('bert-base-chinese')
model_nsp_en = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
model_nsp_cn = BertForNextSentencePrediction.from_pretrained('bert-base-chinese')
tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer_cn = BertTokenizer.from_pretrained('bert-base-chinese')

def predict_mask_cn(text):
    model = model_mask_cn
    tokenizer = tokenizer_cn

    tokenized_text = tokenizer.tokenize(text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    input_ids = torch.tensor([indexed_tokens])

    with torch.no_grad():
        outputs = model(input_ids)

    predictions = outputs[0]
    print(predictions)
    print(predictions[0, tokenized_text.index('[MASK]')])

    predicted_index = torch.argmax(predictions[0, tokenized_text.index('[MASK]')]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token)

    return predicted_token

def predict_mask_en(text):
    model = model_mask_en
    tokenizer = tokenizer_en

    tokenized_text = tokenizer.tokenize(text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    input_ids = torch.tensor([indexed_tokens])

    with torch.no_grad():
        outputs = model(input_ids)

    predictions = outputs[0]
    print(predictions)
    print(predictions[0, tokenized_text.index('[MASK]')])

    predicted_index = torch.argmax(predictions[0, tokenized_text.index('[MASK]')]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token)

    return predicted_token


def predict_nsp_cn(sen1,sen2):
    model = model_nsp_cn
    tokenizer = tokenizer_cn

    tokenized_sen1 = tokenizer.tokenize(sen1)
    tokenized_sen2 = tokenizer.tokenize(sen2)
    tokenized_text = tokenized_sen1 + tokenized_sen2
    print(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids_1 = [0] * len(tokenized_sen1)
    segments_ids_2 = [1] * len(tokenized_sen2)
    segments_ids = segments_ids_1 + segments_ids_2
    segments_tensors = torch.tensor([segments_ids])
    print(segments_tensors)
    with torch.no_grad():
        outputs = model(torch.tensor([indexed_tokens]),token_type_ids=segments_tensors)

    predictions = outputs[0].cpu().numpy()

    result = predictions[0][1] > predictions[0][0]

    print(predictions)
    print(result)
    return result

def predict_nsp_en(sen1,sen2):
    model = model_nsp_en
    tokenizer = tokenizer_en

    tokenized_sen1 = tokenizer.tokenize(sen1)
    tokenized_sen2 = tokenizer.tokenize(sen2)
    tokenized_text = tokenized_sen1 + tokenized_sen2
    print(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids_1 = [0] * len(tokenized_sen1)
    segments_ids_2 = [1] * len(tokenized_sen2)
    segments_ids = segments_ids_1 + segments_ids_2
    segments_tensors = torch.tensor([segments_ids])
    print(segments_tensors)
    with torch.no_grad():
        outputs = model(torch.tensor([indexed_tokens]),token_type_ids=segments_tensors)

    predictions = outputs[0].cpu().numpy()

    result = predictions[0][1] > predictions[0][0]

    print(predictions)
    print(result)
    return result

if __name__ == "__main__":

    sen1 = "Mr. Cassius crossed the highway, and stopped suddenly."
    sen2 = "Something glittered in the nearest red pool before him."
    sen2 = "I borrowed a new book yesterday."

    predict_nsp_en(sen1,sen2)
    exit()

    text = "I want to borrow this new [MASK] in the library"
    # text = "[CLS] Who was Jim Henson ? [SEP] Jim [MASK] was a puppeteer [SEP]"
    predict_mask_en(text,tokenizer,model_path)

    exit()



