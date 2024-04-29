# coding=utf-8

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForNextSentencePrediction
import numpy as np

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

def text_to_emb_cuda(text, tokenizer,model_path):
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    # tokens_tensor = tokens_tensor.to('cuda')
    model = BertModel.from_pretrained(model_path, output_hidden_states=True)
    # model.to('cuda')
    with torch.no_grad():
        outputs = model(tokens_tensor)
    encoded_layers = outputs[0]
    print(encoded_layers)
    em_dict = {}
    for i in range(len(tokenized_text)):
        em_dict[tokenized_text[i]] = encoded_layers[0][i].cpu().numpy()
    return em_dict
    
def cosine(Vec1, Vec2):
    return np.dot(Vec1, Vec2)/(np.linalg.norm(Vec1)*(np.linalg.norm(Vec2)))


def euclidean(Vec1, Vec2):
    return np.linalg.norm(Vec1 - Vec2)

def text_for_mask_lm(text,tokenizer,model_path):

    model = BertForMaskedLM.from_pretrained(model_path)
    model.eval()

    tokenized_text = tokenizer.tokenize(text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    input_ids = torch.tensor([indexed_tokens])

    # segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    # segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        outputs = model(input_ids)

    predictions = outputs[0]
    print(predictions)
    print(predictions[0, tokenized_text.index('[MASK]')])

    predicted_index = torch.argmax(predictions[0, tokenized_text.index('[MASK]')]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token)

def predict_next(sen1,sen2,tokenizer,model_path):
    model = BertForNextSentencePrediction.from_pretrained(model_path)

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


if __name__ == "__main__":
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    # input_ids = torch.tensor(tokenizer.encode("Hello, my dog is [MASK]cute", add_special_tokens=True)).unsqueeze(
    #     0)  # Batch size 1
    # outputs = model(input_ids,masked_lm_labels=input_ids)
    # loss, prediction_scores = outputs[:2]
    # print(prediction_scores)
    # predicted_index = torch.argmax(prediction_scores).item()
    # predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    # print(predicted_token)
    # exit()

    # model_path = r'D:\Projects\BERT\torch_model'
    # # model_path = 'bert-base-uncased'
    # tokenizer = BertTokenizer.from_pretrained(model_path)
    # text = 'I want to borrow this new [MASK] in the library'
    # text_to_emb_cuda(text,tokenizer,model_path)
    # exit()

    # model_path = r'D:\Projects\BERT\torch_model'
    model_path = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    sen1 = "Mr. Cassius crossed the highway, and stopped suddenly."
    sen2 = "Something glittered in the nearest red pool before him."
    sen2 = "I borrowed a new book yesterday."

    predict_next(sen1,sen2,tokenizer,model_path)
    exit()

    # model_path = r'D:\Projects\BERT\torch_model'
    model_path = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_path)

    text = "I want to borrow this new [MASK] in the library"
    # text = "[CLS] Who was Jim Henson ? [SEP] Jim [MASK] was a puppeteer [SEP]"
    text_for_mask_lm(text,tokenizer,model_path)

    exit()



    text1 = "I want to read this new book"
    text2 = "I want to book this room"
    text3 = "there is a book in the corner"

    text1 = "I want to borrow this new book in the library"
    text2 = "I will book 1 room in that hotel"
    text3 = "there is a book in the corner which is very old"

    text1 = "I want to borrow this new book in the library"
    text2 = "I will book 1 room in that hotel"
    text3 = "the book is the only one in this library"
    # text3 = "I looked through the book until I found the right section"

    em_dict_1 = text_to_emb_cuda(text1, tokenizer, model)
    em_dict_2 = text_to_emb_cuda(text2, tokenizer, model)
    em_dict_3 = text_to_emb_cuda(text3, tokenizer, model)

    book1 = em_dict_1['book']
    book2 = em_dict_2['book']
    book3 = em_dict_3['book']

    borrow = em_dict_1['borrow']
    library = em_dict_1['library']

    cosine(book1, book2)
    cosine(book3, book2)
    cosine(book1, book3)
    book1[:20]
    book2[:20]
    book3[:20]

    print('cosine of book1, book2 :', cosine(book1, book2))
    print('cosine of book2, book3 :', cosine(book3, book2))
    print('cosine of book1, book3 :', cosine(book1, book3))

    text1 = 'The handle helps you open the drawer'
    text2 = 'A secretary can handle office communications'

    em_dict_1 = text_to_emb_cuda(text1, tokenizer, model)
    em_dict_2 = text_to_emb_cuda(text2, tokenizer, model)
    book1 = em_dict_1['handle']
    book2 = em_dict_2['handle']
    print('cosine of book1, book2 :', cosine(book1, book2))

    text = "I know who he is"
    em_dict = text_to_emb_cuda(text, tokenizer, model)
    for key,value in em_dict.items():
        print(key)
        print(value)
    




# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained('/home/leo/lh/torch_models/bert_base')

# # Tokenize input
# text = "[CLS] who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
# tokenized_text = tokenizer.tokenize(text)

# # Mask a token that we will try to predict back with `BertForMaskedLM`
# masked_index = 8
# tokenized_text[masked_index] = '[MASK]'
# assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

# # Convert token to vocabulary indices
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
# segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[CLS] I know who he is [SEP]"))
# segments_ids = [0, 0, 0, 0, 0, 0, 0]

# # Convert inputs to PyTorch tensors
# tokens_tensor = torch.tensor([indexed_tokens])
# segments_tensors = torch.tensor([segments_ids])

# # Load pre-trained model (weights)
# model = BertModel.from_pretrained('/home/leo/lh/torch_models/bert_base')

# # Set the model in evaluation mode to desactivate the DropOut modules
# # This is IMPORTANT to have reproductible results during evaluation!
# model.eval()

# # If you have a GPU, put everything on cuda
# tokens_tensor = tokens_tensor.to('cuda')
# segments_tensors = segments_tensors.to('cuda')
# model.to('cuda')

# # Predict hidden states features for each layer
# with torch.no_grad():
#     # See the models docstrings for the detail of the inputs
#     outputs = model(tokens_tensor, token_type_ids=segments_tensors)
#     # PyTorch-Transformers models always output tuples.
#     # See the models docstrings for the detail of all the outputs
#     # In our case, the first element is the hidden state of the last layer of the Bert model
#     encoded_layers = outputs[0]
# # We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
# # assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)
# print(encoded_layers)
# print(encoded_layers.shape)
# print(outputs)
# print(len(outputs))

# # Load pre-trained model (weights)
# model = BertForMaskedLM.from_pretrained('bert-base-uncased')
# model.eval()

# # If you have a GPU, put everything on cuda
# tokens_tensor = tokens_tensor.to('cuda')
# segments_tensors = segments_tensors.to('cuda')
# model.to('cuda')

# # Predict all tokens
# with torch.no_grad():
#     outputs = model(tokens_tensor, token_type_ids=segments_tensors)
#     predictions = outputs[0]

# # confirm we were able to predict 'henson'
# predicted_index = torch.argmax(predictions[0, masked_index]).item()
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
# assert predicted_token == 'henson'


# text = 'China is one of the biggest country in the world'

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids)
# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple