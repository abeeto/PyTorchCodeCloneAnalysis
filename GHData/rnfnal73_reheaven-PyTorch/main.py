import os
import torch
import sys
import transformers
from my_transformers import(
    ElectraConfig,
    ElectraForSequenceClassification,
    ElectraTokenizer,
)
file_path = os.path.dirname(os.path.realpath(__file__))

labels = ['0', '1']

config = ElectraConfig.from_pretrained(
            file_path+r"/wordpiece_base",
            num_labels=2,
            id2label={str(i): label for i, label in enumerate(labels)},
            label2id={label: i for i, label in enumerate(labels)},
            )

model = ElectraForSequenceClassification.from_pretrained(
            file_path+r"/wordpiece_base",
            from_tf=True,
            config=config
            )
loaded = torch.load(file_path+r"/pytorch_model.bin",map_location='cpu')  #torch.load(PATH,map_location), map_location은 model을 load하는 device 종류
model.load_state_dict(loaded,strict=False)

#tokenizer = ElectraTokenizer.from_pretrained(file_path+r"/wordpiece_base")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-discriminator")

while True:
	sentence = input()
	t_sen = tokenizer.tokenize(sentence)
	print('t_sen:',t_sen)
	g_sen = tokenizer.encode(t_sen,return_tensors="pt")
	print('g_sen:',g_sen)
	model_input = g_sen
	classification_result = model(model_input)
	print('result',classification_result)
	result = torch.argmax(classification_result[0])
	print('결과: ',result.item())

print('\ndone')