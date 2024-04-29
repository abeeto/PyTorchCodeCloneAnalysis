import torch
import torch.nn as nn
import torch.nn.functional as F

class Memory(nn.Module):
    """
    Reference: github.com/seoungwugoh/STM
    m_in, m_out: memory keys and values
    q_in, q_out: query keys and values
    """
    def __init__(self):
        super(Memory, self).__init__()
 
    def forward(self, m_in, m_out, q_in, q_out):
        Bq, D_e, Hq, Wq = q_in.size()
        T,  D_e, Hm, Wm = m_in.size()
        _,  D_o, _,  _  = m_out.size()

        # Keys: (T * Hm * Wm, De)
        m_in = m_in.transpose(0, 1)
        mi = m_in.reshape(D_e, T * Hm * Wm)
        mi = torch.transpose(mi, 0, 1)       

        # Queries: (Bq, De, Hq * Wq)
        qi = q_in.reshape(Bq, D_e, Hq * Wq)  

        # Activations: (Bq, T * Hm * Wm, Hq * Wq)
        p = torch.matmul(mi, qi)             
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1)

        # Values: (T * Hm * Wm, Do) = (N, Do)
        m_out = m_out.transpose(0, 1)
        mo = m_out.reshape(D_o, T * Hm * Wm) 

        # Weighted sum of values: 
        # (Do, N) X (Bq, N, Hq * Wq) = (Bq, Do, Hq * Wq) 
        mem = torch.matmul(mo, p)            
        mem = mem.reshape(Bq, D_o, Hq, Wq)

        mem_out = torch.cat([mem, q_out], dim=1)
        return mem_out, p
