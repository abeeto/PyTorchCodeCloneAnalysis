"""
在此处定义所有模型
"""

from logging import raiseExceptions
import torch
import torch.nn as nn
from transformers import BertModel
from config import *
from utils import *

# 返回向量最大值位置对应的索引 
def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

# 数值稳定版本的LogSumExp，可参考：https://blog.csdn.net/liyu0611/article/details/100547145
def log_sum_exp_batch(log_Tensor, axis=-1):
    return torch.max(log_Tensor, axis)[0] + \
        torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0],-1,1)).sum(axis))

# CRF 模型，可参考：https://zhuanlan.zhihu.com/p/97829287，讲的很详细
class CRF(nn.Module):
    def __init__(self):
        super(CRF, self).__init__()
        # 标签转移矩阵，transitions[i,j]为标签j转移至标签i的分数
        self.transitions = nn.Parameter(torch.randn(
            tagset_size, tagset_size
        ))
        
        # 将所有标签到开始标签的转移分数设置得很低
        self.transitions.data[start_label_id, :] = -10000
        # 将结束标签到开始标签的转移分数设置得很低
        self.transitions.data[:, end_label_id] = -10000
        
    # CRF计算给定特征（发射矩阵，即每一个时刻属于各标签的概率，可以由bert、lstm等求得），求条件概率log部分的分数
    # 使用动态规划求解，可参考https://zhuanlan.zhihu.com/p/97829287
    # feats 每个时刻输出为各个标签的概率
    # masks 输入掩码
    def _forward_alg(self, feats, masks):
        T = feats.shape[1]  
        batch_size = feats.shape[0]
        
        log_alpha = torch.Tensor(batch_size, 1, tagset_size).fill_(-10000.).to(device)  #[batch_size, 1, 16]
        log_alpha[:, 0, start_label_id] = 0
        
        for t in range(1, T):
            mask = masks[:,t].unsqueeze(1) # batch_size * 1
            log_alpha = ((log_sum_exp_batch(self.transitions + log_alpha, axis=-1) \
                         + feats[:, t]) * mask + log_alpha.squeeze(1) * (1-mask)).unsqueeze(1)

        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    # CRF计算给定特征（发射矩阵，即每一个时刻属于各标签的概率，可以由bert、lstm等求得），求条件概率除去log部分的分数
    # 可参考https://zhuanlan.zhihu.com/p/97829287
    # feats 每个时刻输出为各个标签的概率
    # tags 每个样本的真实标签序列
    # masks 输入掩码
    def _score_sentence(self, feats, tags, masks):
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size,tagset_size,tagset_size)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0],1)).to(device)
        for t in range(1, T):
            mask = masks[:,t].unsqueeze(1) # batch_size * 1
            score = score + \
                (batch_transitions.gather(-1, (tags[:, t]*tagset_size+tags[:, t-1]).view(-1,1)) \
                    + feats[:, t].gather(-1, tags[:, t].view(-1,1)).view(-1,1)) * mask
        return score

    # CRF优化损失函数，由前述 _forward_alg 函数与 _score_sentence 函数相加得到
    # feats 每个时刻输出为各个标签的概率
    # tags 每个样本的真实标签序列
    # masks 输入掩码
    def neg_log_likelihood(self, feats, tags, masks):
        forward_score = self._forward_alg(feats, masks)
        gold_score = self._score_sentence(feats, tags, masks)
        return torch.mean(forward_score - gold_score)

    # 使用维特比算法根据给定特征（发射矩阵，即每一个时刻属于各标签的概率，可以由bert、lstm等求得）求最大概率路径
    # feats 每个时刻输出为各个标签的概率
    def _viterbi_decode(self, feats, masks):
        
        # 获取序列最大长度以及批大小
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # 使用维特比算法求最优路径，维特比算法使用动态规划思想，可参考 https://www.zhihu.com/question/20136144
        # log_delta 保存当前时刻到各个标签的最优路径的得分
        log_delta = torch.Tensor(batch_size, 1, tagset_size).fill_(-10000.).to(device)
        log_delta[:, 0, start_label_id] = 0.   # 初始化第0个时刻到 START 标签概率最大
        
        # psi保存各个时刻到各标签的得分最大路径的上一个时刻的标签，方便进行回溯
        psi = torch.zeros((batch_size, T, tagset_size), dtype=torch.long).to(device)  # psi[0]=0000 useless
        for t in range(1, T):
            mask = masks[:,t].unsqueeze(1) # batch_size * 1
            a,b = torch.max(self.transitions + log_delta, -1)
            log_delta = ((a + feats[:, t]) * mask + log_delta.squeeze(1) * (1-mask)).unsqueeze(1)
            psi[:, t] = b * mask.long() + \
                 torch.from_numpy(np.arange(tagset_size)).long().view(1,-1).to(device) * (1 - mask).long()
            # log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            # log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # 回溯路径
        path = torch.zeros((batch_size, T), dtype=torch.long).to(device)

        # 最后一个时刻得分最高的标签
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T-2, -1, -1):
            # 每一个时刻
            path[:, t] = psi[:, t+1].gather(-1,path[:, t+1].view(-1,1)).squeeze()

        return max_logLL_allz_allx, path

# BERT+LSTM+CRF进行序列标注的模型，LSTM为可选
class BertLstmCRF(nn.Module):
    def __init__(self):
        super(BertLstmCRF, self).__init__()

        # bert模型，encoder模型，可以换成其他的，换了之后把下面的bert_output_size也改了就行了
        self.bert = BertModel.from_pretrained(bert_model)
        # 双向LSTM模型，接在bert后，可选，768为bert输出维度
        bert_output_size = 768
        self.lstm = nn.LSTM(bidirectional=True, num_layers=1, input_size=bert_output_size, \
                         hidden_size=lstm_hidden_size, batch_first=True)
        # 全连接层，将bert或bert+lstm编码的特征转化为tagset_size维度
        if use_lstm:
            self.fc = nn.Linear(lstm_hidden_size*2, tagset_size)
        else:
            self.fc = nn.Linear(bert_output_size, tagset_size)
        nn.init.kaiming_normal_(self.fc.weight)
        # CRF模型
        self.crf = CRF()

        # Dropout，随机失活缓解过拟合，一般加在全连接后面，这里加载fc层后面，bert输出层、lstm输出层等位置也可以考虑加上
        # 默认值为0即不进行dropout，如需实验加了的效果可修改config.py里面的dropout参数
        self.dropout = nn.Dropout(dropout)

        # tanh激活函数
        self.act = nn.Tanh()

    # 前向计算，训练时返回损失，测试时返回最优路径
    # sentences 输入句子
    # masks 掩码，真实输入处为1，pad处为0
    # tags 标签
    # traing 是否训练
    def forward(self, sentences, masks, tags=None, training=True): 
        # 改了encoder模型后这里也改一下
        enc, _  = self.bert(sentences)
        if use_lstm:
            enc, _ = self.lstm(enc)
        feats = self.fc(enc)
        feats = self.dropout(feats)

        if training:
            # 训练但是没有提供标签
            if tags is None:
                raise Exception('Traing but no tags provided.')
            return self.crf.neg_log_likelihood(feats, tags, masks)
        else:
            score, tag_seq = self.crf._viterbi_decode(feats, masks)
            return score, tag_seq