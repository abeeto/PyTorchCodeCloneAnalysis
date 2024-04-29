import torch
import torchvision.transforms.functional as TF
# Moore neighborhood
Q = 3

def _lucas_kanade_method(Ix_patches, Iy_patches, It_patches, ix, iy):
    p_Ix = Ix_patches[iy, ix].ravel()
    p_Iy = Iy_patches[iy, ix].ravel()
    p_It = It_patches[iy, ix].ravel().unsqueeze(1)
    A = torch.stack([p_Ix, p_Iy]).T
    b = p_It
    v = torch.linalg.pinv(A) @ b
    return v.squeeze(1)

def lucas_kanade_method(img0, img1):
    It, Iy, Ix = torch.gradient(torch.vstack((img0, img1)))
    It, Iy, Ix = It[0], Iy[0], Ix[0]
    Ix_patches = Ix.unfold(0, Q, Q).unfold(1, Q, Q)
    Iy_patches = Iy.unfold(0, Q, Q).unfold(1, Q, Q)
    It_patches = It.unfold(0, Q, Q).unfold(1, Q, Q)
    
    return torch.stack([
        torch.stack([
             _lucas_kanade_method(Ix_patches, Iy_patches, It_patches, ix, iy) for ix in range(Ix_patches.size(1))
        ])
        for iy in range(Ix_patches.size(0))
    ])

def optical_flow(img0, img1):
    N = torch.log2(
        torch.tensor(img0.size(1) if img0.size(1) < img0.size(2) else img0.size(2))
    ).int() - torch.log2(torch.tensor(8)).int()
    pyramid = [
        lucas_kanade_method(
            TF.resize(img0, (img0.size(1)//r, img0.size(2)//r)),
            TF.resize(img1, (img1.size(1)//r, img1.size(2)//r))
        ).permute(2, 0, 1) * r * Q
        for r in reversed([2 ** i for i in range(N)])
    ]

    return torch.stack([
        torch.nn.Upsample(size=img0.size()[1:], mode='nearest')(p.unsqueeze(0)).squeeze(0).permute(1, 2, 0)
        for n, p in enumerate(reversed(pyramid))
    ]).mean(dim=0)