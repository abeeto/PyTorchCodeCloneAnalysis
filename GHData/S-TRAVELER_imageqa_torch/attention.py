import torch
import torch.nn.functional as F

nn=torch.nn

class Attention(torch.nn.Module):
    def __init__(self, options):
        self._dropout=options['use_attention_drop']
        self._drop_ratio=options['drop_ratio']
        super(Attention,self).__init__()
        self.image_att_mlp=nn.Sequential(
            nn.Linear(1024,512),
            nn.Tanh()
        )
        self.sent_att_mlp=nn.Sequential(
            nn.Linear(1024,512),
            nn.Tanh()
        )
        self.combined_att_mlp=nn.Sequential(
            nn.Linear(512,1),
            nn.Tanh()
        )

    def forward(self, image_emb, h_encode):
        image_feat_att=self.image_att_mlp(image_emb) #(100,196,1024)=>(100,196,512)
        h_encode_att=self.sent_att_mlp(h_encode) #(100,1,1024)=>(100,1,512)
        combined_feat_att=image_feat_att+ \
                        h_encode_att[:,None,:] #(100,196,512)+(100,1,512)=>(100,196,512)
        
        if self._dropout:
            combined_feat_att=nn.Dropout(combined_feat_att, self._drop_ratio) #(100,196,512)
        
        combined_feat_att=self.combined_att_mlp(combined_feat_att) #(100,196,512)=>(100,196,1)
        prob_attention=nn.Softmax(combined_feat_att[:,:,0])        #(100,196)

        #(100,196,1)*(100,196,1024)=>(100,196,1024)=>(100,1,1024)
        image_feat_ave=(prob_attention[:, :, None] * image_emb).sum(dim=1) 
        
        combined_out=image_feat_ave+h_encode
        return combined_out #(100,1,1024)
