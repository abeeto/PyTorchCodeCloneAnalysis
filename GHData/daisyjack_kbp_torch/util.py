# coding: utf-8
import torch

def switch(vec1, vec2, mask):
    """
    switch function for pytorch

    args:
        vec1 (any size) : input tensor corresponding to 0
        vec2 (same to vec1) : input tensor corresponding to 1
        mask (same to vec1) : input tensor, each element equals to 0/1
    return:
        vec (*)
    """
    catvec = torch.cat([vec1.view(-1, 1), vec2.view(-1, 1)], dim=1)
    switched_vec = torch.gather(catvec, 1, mask.long().view(-1, 1))
    return switched_vec.view(-1)