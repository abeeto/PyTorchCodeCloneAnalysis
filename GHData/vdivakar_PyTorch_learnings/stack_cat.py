''' Concatenate -> adding to existing axis
    Stack       -> creating a new axis'''

import torch

img1 = torch.rand(3,120,120)
img2 = torch.rand(3,120,120)
img3 = torch.rand(3,120,120)
img4 = torch.rand(3,120,120)

'''Creating a batch from images'''
batch = torch.stack((img1,img2,img3,img4), axis=0)
print("batch  ", batch.shape) #[4,3,120,120]

'''Adding imgs to existing batch'''
batch2 = torch.cat((batch, img1.unsqueeze(0)), axis=0)
print("batch2 ", batch2.shape) #[5,3,120,120]

batch3 = torch.cat(
    (
        batch,
        torch.stack(
            (
                img1, img2, img3
            ), axis=0
        )
    ),axis=0
)
print("batch3 ", batch3.shape) #[7,3,120,120]

'''Adding 2 batches'''
final_batch = torch.cat((batch, batch2), axis=0)
print("final  ", final_batch.shape) #[9,3,120,120]