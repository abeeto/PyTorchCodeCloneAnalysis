import torch
import torch.nn as nn

class BertClassifical(nn.Module):

    def __init__(self, config,BertModel,pretrained_model_path,layer_output_counts = 0,):
        super(BertClassifical, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_output_counts = layer_output_counts
        if self.layer_output_counts:
            self.dynamic_layers = [nn.Linear(config.hidden_size,1) for i in range(self.layer_output_counts)]
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.classifical = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重
        nn.init.xavier_normal_(self.classifical.weight)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None):
        # Forward pass through pre-trained BERT
        outputs = self.bert(input_ids,
                            position_ids=position_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            head_mask=head_mask,
                            output_hidden_states=True)
        #这里outputs输出的有最后一层隐状态结果,池化结果,所有层的隐状态的输出
        if self.layer_output_counts:
            all_hidden_states = outputs[-1][::-1]
            #print("输出的层数为{}".format(len(all_hidden_states)))
            self.all_layer_output = all_hidden_states[0].cuda()*self.dynamic_layers[0].cuda()(all_hidden_states[0].cuda())
            for i in range(1,self.layer_output_counts):
                self.all_layer_output += all_hidden_states[i].cuda()*self.dynamic_layers[i].cuda()(all_hidden_states[i].cuda())
            #print(self.all_layer_output.size())
            #x = self.all_layer_output.permute(0, 2, 1)
            #finally_out = self.maxpool(x)
            #print(finally_out.size())
            #finally_out = finally_out.squeeze(dim=-1)
            finally_out = self.all_layer_output[:,0,:]
            return self.classifical(finally_out)
        else:
            #pooled_output = self.dropout(pooled_output)
            pooled_output = outputs[1]
            return self.classifical(pooled_output)

