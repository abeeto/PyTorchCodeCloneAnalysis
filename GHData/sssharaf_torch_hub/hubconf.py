dependencies = ['torch','transformers']
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig,BertForMaskedLM,BertModel,DistilBertTokenizer, DistilBertModel,DistilBertForSequenceClassification

def model7(*args, **kwargs):
    model =MyModel7()
    checkpoint = 'https://s-ml-pretrained.s3.amazonaws.com/model-7.dat'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint,map_location=torch.device('cpu'), progress=True))
    return model

def model5(*args, **kwargs):
    model =MyModel5()
    checkpoint = 'https://s-ml-pretrained.s3.amazonaws.com/model-5.dat'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint,map_location=torch.device('cpu'), progress=True))
    return model

def model4(*args, **kwargs):
    model =MyModel4()
    checkpoint = 'https://s-ml-pretrained.s3.amazonaws.com/model-4.dat'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint,map_location=torch.device('cpu'), progress=True))
    return model

def model31(*args, **kwargs):
    model =MyModel3()
    checkpoint = 'https://s-ml-pretrained.s3.amazonaws.com/model-3-1.dat'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint,map_location=torch.device('cpu'), progress=True))
    return model

def model3(*args, **kwargs):
    model =MyModel3()
    checkpoint = 'https://s-ml-pretrained.s3.amazonaws.com/model-3.dat'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint,map_location=torch.device('cpu'), progress=True))
    return model

def model2(*args, **kwargs):
    model =MyModel2()
    checkpoint = 'https://s-ml-pretrained.s3.amazonaws.com/model-2.dat'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint,map_location=torch.device('cpu'), progress=True))
    return model

def model11(*args, **kwargs):
    model =MyModel11()
    checkpoint = 'https://s-ml-pretrained.s3.amazonaws.com/model-11.dat'
    device = 'cpu' if 'device' not in kwargs else kwargs['device']
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint,map_location=torch.device(device), progress=True))
    return model


############################### Model 11  ############################################
# Model
class MyModel11(nn.Module):
    def __init__(self, freeze_bert=True, attn_dropout=0.3):
        super().__init__()
        self.model_version = '11'
        # MODEL_NAME='bert-base-uncased'
        self.bert_lyr = BertForSequenceClassification(BertConfig.from_pretrained('bert-base-uncased',num_labels=9))

    def freeze_bert(self):
        self._freeze_bert(self.bert_lyr)

    def _freeze_bert(self, bert_model):
        for p in bert_model.bert.parameters():
            p.requires_grad = False

    def forward(self, seq, attn_masks, output_attn=False, output_hs=False):
        (o,) = self.bert_lyr(seq, attn_masks)
        return (o[:, :4], o[:, 4:])


############################### Model 3-1  ############################################
class MyModel31(nn.Module):
    def __init__(self, freeze_bert=True):
        super().__init__()
        self.model_version = 3

        self.bert_lyr = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

        self.a_attn = nn.Linear(768, 1, bias=False)

        self.c_attn = nn.Linear(768, 1, bias=False)

        self.attn_dropout = nn.Dropout(0.1, inplace=False)

        self.a_switch = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768 * 2, 64),
            nn.ELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.c_switch = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768 * 2, 64),
            nn.ELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.action_cls_lyr = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 64, bias=False),
            nn.ELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 4, bias=False),
        )

        self.component_cls_lyr = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 64, bias=False),
            nn.ELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 5, bias=False),
        )

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_lyr.parameters():  # self.bert_lyr.parameters():
                p.requires_grad = False
        # nn.init.xavier_uniform_(self.action_cls_lyr)
        # nn.init.xavier_uniform_(self.component_cls_lyr)

    def forward(self, seq, attn_masks, output_attn=False, output_hs=False, output_switch=False):
        attn_mask_cls = (1 - attn_masks) * -10000
        attn_mask_cls.unsqueeze_(dim=-1)

        seq_emb, ctx, hs = self.bert_lyr(seq, attention_mask=attn_masks)

        a = self.a_attn(seq_emb)
        a = a + attn_mask_cls
        a = a_output = a.softmax(dim=1)
        a = self.attn_dropout(a)
        a = torch.mul(seq_emb, a)
        a = a.mean(dim=1)

        c = self.c_attn(seq_emb)
        c = c + attn_mask_cls
        c = c_output = c.softmax(dim=1)
        c = self.attn_dropout(c)
        c = torch.mul(seq_emb, c)
        c = c.mean(dim=1)

        a_switch = self.a_switch(torch.cat([ctx.detach(), a], dim=-1))
        a = (1 - a_switch) * a + a_switch * ctx

        c_switch = self.c_switch(torch.cat([ctx.detach(), c], dim=-1))
        c = (1 - c_switch) * c + c_switch * ctx

        outputs = [self.action_cls_lyr(a), self.component_cls_lyr(c)]
        if (output_attn):
            outputs += [a_output, c_output]
        if output_hs:
            outputs += [hs]
        if output_switch:
            outputs += [a_switch, c_switch]
        return outputs

############################### Model 3  ############################################
class MyModel3(nn.Module):
    def __init__(self, freeze_bert=True):
        super().__init__()
        self.model_version = 3

        self.bert_lyr = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

        self.a_attn = nn.Linear(768, 1)

        self.c_attn = nn.Linear(768, 1)

        self.attn_dropout = nn.Dropout(0.1, inplace=False)

        # self.ctx_transfomer = nn.Sequential(nn.LayerNorm(768),nn.Dropout(0.1),nn.Linear(768,768),nn.Tanh())

        self.a_switch = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768 * 2, 1),
            # nn.LayerNorm(1),
            nn.Sigmoid()
        )

        self.c_switch = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768 * 2, 1),
            # nn.LayerNorm(1),
            nn.Sigmoid(),
        )

        self.action_cls_lyr = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 64, bias=False),
            nn.ELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 4, bias=False),
            # nn.LayerNorm(len(action_le.classes_)),
            # nn.Linear(768,len(action_le.classes_),bias=False),
        )

        self.component_cls_lyr = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 64, bias=False),
            nn.ELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 5, bias=False),
            # nn.LayerNorm(len(component_le.classes_)),
            # nn.Linear(768,len(component_le.classes_),bias=False),
        )

        # Freeze bert layers
        if freeze_bert:
            for lyr in self.bert_lyr.encoder.layer[:-2]:
                for p in lyr.parameters():  # self.bert_lyr.parameters():
                    p.requires_grad = False
        # nn.init.xavier_uniform_(self.action_cls_lyr)
        # nn.init.xavier_uniform_(self.component_cls_lyr)

    def forward(self, seq, attn_masks, output_attn=False, output_hs=False, output_switch=False):
        attn_mask_cls = (1 - attn_masks) * -10000
        attn_mask_cls.unsqueeze_(dim=-1)

        seq_emb, ctx, hs = self.bert_lyr(seq, attention_mask=attn_masks)
        # ctx = self.ctx_transfomer(ctx)
        a = self.a_attn(seq_emb)
        a = a + attn_mask_cls
        a = a_output = a.softmax(dim=1)
        # a = self.attn_dropout(a)
        a = torch.mul(seq_emb, a)
        a = a.mean(dim=1)

        c = self.c_attn(seq_emb)
        c = c + attn_mask_cls
        c = c_output = c.softmax(dim=1)
        c = self.attn_dropout(c)
        c = torch.mul(seq_emb, c)
        c = c.mean(dim=1)

        a_switch = self.a_switch(torch.cat([ctx, a], dim=-1))
        a = a_switch * a + (1.0 - a_switch) * ctx

        c_switch = self.c_switch(torch.cat([ctx, c], dim=-1))
        c = c_switch * c + (1.0 - c_switch) * ctx

        outputs = [self.action_cls_lyr(a), self.component_cls_lyr(c)]
        if (output_attn):
            outputs += [a_output, c_output]
        if output_hs:
            outputs += [hs]
        if output_switch:
            outputs += [a_switch, c_switch]
        return outputs


############################### Model 2  ############################################

class MyModel2(nn.Module):
    def __init__(self, freeze_bert=True):
        super().__init__()
        self.model_version = 2 - 1
        # self.static_bert_lyr = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=False)
        self.bert_lyr = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

        self.a_attn = nn.Linear(768, 1)

        self.c_attn = nn.Linear(768, 1)

        self.action_cls_lyr = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 64, bias=True),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, 4, bias=True),
            nn.LayerNorm(4),
            # nn.Linear(768,len(action_le.classes_),bias=False),
        )

        self.component_cls_lyr = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 64, bias=True),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, 5, bias=True),
            nn.LayerNorm(5),
            # nn.Linear(768,len(component_le.classes_),bias=False),
        )

        # for p in self.static_bert_lyr.parameters():
        #   p.requires_grad = False

        # Freeze bert layers
        if freeze_bert:
            for lyr in self.bert_lyr.encoder.layer[:-2]:
                for p in lyr.parameters():  # self.bert_lyr.parameters():
                    p.requires_grad = False
        # nn.init.xavier_uniform_(self.action_cls_lyr)
        # nn.init.xavier_uniform_(self.component_cls_lyr)

    def forward(self, seq, attn_masks, output_attn=False, output_hs=False):
        attn_mask_cls = (1 - attn_masks) * -10000
        attn_mask_cls.unsqueeze_(dim=-1)

        # static_emb,static_ctx = self.static_bert_lyr(seq,attention_mask =attn_masks)
        seq_emb, ctx, hs = self.bert_lyr(seq, attention_mask=attn_masks)
        # seq_emb +=static_emb
        a = self.a_attn(seq_emb)
        a = a + attn_mask_cls
        a = a_output = a.softmax(dim=1)
        a = torch.mul(seq_emb, a)
        a = a.mean(dim=1)

        c = self.c_attn(seq_emb)
        c = c + attn_mask_cls
        c = c_output = c.softmax(dim=1)
        c = torch.mul(seq_emb, c)
        c = c.mean(dim=1)

        outputs = [self.action_cls_lyr(a), self.component_cls_lyr(c)]
        if (output_attn):
            outputs += [a_output, c_output]
        if output_hs:
            outputs += [hs]
        return outputs
#####################################################################################


############################### Model 4  ############################################
class MyModel4(nn.Module):
    def __init__(self, freeze_bert=True):
        super().__init__()
        self.model_version = 4

        self.bert_lyr = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True,
                                                  output_attentions=True)

        # self.comp_bert_lyr = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True,output_attentions=True)

        self.config = self.bert_lyr.config;

        self.action_cls_lyr = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 4),
        )
        self.comp_cls_lyr = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 5),
        )

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_lyr.parameters():
                p.requires_grad = False
            for p in self.comp_bert_lyr.parameters():
                p.requires_grad = False

    def forward(self, seq, attn_masks, output_attn=False, output_hs=False):

        seq_emb, pooled, hs, attn = self.bert_lyr(seq, attention_mask=attn_masks)

        # c_seq_emb,c_pooled,c_hs,c_attn = self.comp_bert_lyr(seq,attention_mask =attn_masks)

        outputs = []
        outputs += [
            self.action_cls_lyr(pooled),
            self.comp_cls_lyr(pooled),
        ]
        return outputs


############################### Model 5  ############################################
class MyModel5(nn.Module):
    def __init__(self, freeze_bert=True):
        super().__init__()
        self.model_version = 5

        self.bert_lyr = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True,
                                                  output_attentions=True)

        self.comp_bert_lyr = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True,
                                                       output_attentions=True)

        self.config = self.bert_lyr.config;

        self.action_cls_lyr = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 4),
        )
        self.comp_cls_lyr = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 5),
        )

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_lyr.parameters():
                p.requires_grad = False
            for p in self.comp_bert_lyr.parameters():
                p.requires_grad = False

    def forward(self, seq, attn_masks, output_attn=False, output_hs=False):

        seq_emb, pooled, hs, attn = self.bert_lyr(seq, attention_mask=attn_masks)

        c_seq_emb, c_pooled, c_hs, c_attn = self.comp_bert_lyr(seq, attention_mask=attn_masks)

        outputs = []
        outputs += [
            self.action_cls_lyr(pooled),
            self.comp_cls_lyr(c_pooled),
        ]
        return outputs


############################### Model 7  ############################################

class MyModel7(nn.Module):
    def __init__(self, freeze_bert=True):
        super().__init__()
        self.model_version = 7

        self.bert_lyr = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True,
                                                  output_attentions=True)

        # self.comp_bert_lyr = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True,output_attentions=True)

        self.config = self.bert_lyr.config;

        self.a_attn = nn.Linear(768, 1)

        self.c_attn = nn.Linear(768, 1)

        self.action_cls_lyr = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 4),
        )
        self.comp_cls_lyr = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 5),
        )

        # Freeze bert layers
        if freeze_bert:
            self.freeze_bert()
        else:
            self.freeze_bert()
            self.unfreeze_bert()

    def freeze_bert(self):
        for p in self.bert_lyr.parameters():
            p.requires_grad = False
        # for p in self.comp_bert_lyr.parameters():
        #   p.requires_grad = False

    def unfreeze_bert(self, from_lyr=6):

        for lyr in self.bert_lyr.encoder.layer[-6:]:
            for p in lyr.parameters():
                p.requires_grad = True
        # for lyr in self.comp_bert_lyr.encoder.layer[-6:]:
        #   for p in lyr.parameters():
        #     p.requires_grad = True

    def forward(self, seq, attn_masks, output_attn=False, output_hs=False):
        attn_mask_cls = (1 - attn_masks) * -10000
        attn_mask_cls.unsqueeze_(dim=-1)
        seq_emb, pooled, hs, attn = self.bert_lyr(seq, attention_mask=attn_masks)

        # c_seq_emb,c_pooled,c_hs,c_attn = self.comp_bert_lyr(seq,attention_mask =attn_masks)
        a, a_output = self.attention(seq_emb, self.a_attn, attn_mask_cls)

        c, c_output = self.attention(seq_emb, self.c_attn, attn_mask_cls)

        a_pooled = a
        c_pooled = c
        outputs = []
        outputs += [
            self.action_cls_lyr(a_pooled),
            self.comp_cls_lyr(c_pooled),
        ]
        if (output_attn):
            outputs += [a_output, c_output]
        return outputs

    def attention(self, seq_emb, attn_lyr, attn_mask_cls):
        a = attn_lyr(seq_emb)
        a = a + attn_mask_cls
        a = a_output = a.softmax(dim=1)
        # a_output = a.clone()
        a = torch.mul(seq_emb, a)
        a = a.mean(dim=1)
        return a, a_output
