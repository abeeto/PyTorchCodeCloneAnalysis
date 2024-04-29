import argparse
import pickle
import codecs

import numpy as np
import torch

from com.preprocess import transform_data_to_id
from com.utils import pad_answer, padding, pad_wrong_answer

parser = argparse.ArgumentParser(description='inference procedure, note you should train the data at first')


parser.add_argument('--word_path', type=str, default='data/word2id.obj',
                    help='location of the word2id.obj')

parser.add_argument('--output', type=str, default='data/5th-submita.txt', # TODO: 更换提交文本名称
                    help='prediction path')
parser.add_argument('--model', type=str, default='net/dcn.pt',# TODO: 更换模型
                    help='model path')


args = parser.parse_args()

# TODO: 通过更换model_name来切换模型
model_name='dcn'
# model_name='mwan_f0'
# model_name='mwan_f1'
# model_name='mwan_o'
model_path="net/"+model_name+".pt"
# TODO: 通过更换dt_name来更换数据集
dt_name="testa"
# dt_name="dev"
# dt_name="train"

with open(model_path, 'rb') as f:
    model = torch.load(f)
if torch.cuda.is_available():
    model.cuda()

with open(args.word_path, 'rb') as f:
    word2id = pickle.load(f)

# raw_data = seg_data(args.data)
raw_data=pickle.load(open("data/"+dt_name+"_seg.pkl","rb")) # TODO: 更改预测数据
transformed_data = transform_data_to_id(raw_data, word2id)
data = [x + [y[2]] for x, y in zip(transformed_data, raw_data)]
data = sorted(data, key=lambda x: len(x[1]))
print ('test data size {:d}'.format(len(data)))


def inference():
    model.eval()
    predictions = []
    id_prediction={}
    with torch.no_grad():
        for i in range(0, len(data), args.batch_size):
            print("{} in {}".format(i,len(data)))
            one = data[i:i + args.batch_size]
            query, _ = padding([x[0] for x in one], max_len=opts["q_len"])
            passage, _ = padding([x[1] for x in one], max_len=opts["p_len"])
            answer = pad_answer([x[2] for x in one],max_len=opts["alt_len"])
            str_words = [x[-1] for x in one]
            ids = [x[3] for x in one]
            answer=pad_wrong_answer(answer)
            query=torch.LongTensor(query)
            passage = torch.LongTensor(passage)
            #print(np.shape(answer))
            answer=torch.LongTensor(answer)
            if torch.cuda.is_available():
                query = query.cuda()
                passage = passage.cuda()
                answer = answer.cuda()
            output = model([query, passage, answer,ids,False,False])
            for q_id, prediction, candidates in zip(ids, output, str_words):
                id_prediction[q_id]=int(prediction)
                prediction_answer = u''.join(candidates[prediction])
                predictions.append(str(q_id) + '\t' + prediction_answer)
    outputs = u'\n'.join(predictions)
    with codecs.open(args.output, 'w',encoding='utf-8') as f:
        f.write(outputs)
    with open("pkl_records/"+model_name+"."+dt_name+".pkl","wb") as f: # TODO: 更换pkl文件名称
        pickle.dump(id_prediction,f)
    print('done!')


if __name__ == '__main__':
    opts = json.load(open("models/config.json"))
    inference()