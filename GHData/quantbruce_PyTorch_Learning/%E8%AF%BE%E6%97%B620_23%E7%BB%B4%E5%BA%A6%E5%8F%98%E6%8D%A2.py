import torch

# a = torch.rand([4, 1, 28, 28])
# print(a.shape)

# print(a.unsqueeze(0).shape)
# print(a.unsqueeze(-1).shape)
# print(a.unsqueeze(4).shape)
# print(a.unsqueeze(-4).shape)
# print(a.unsqueeze(-5).shape)
# print(a.unsqueeze(5).shape)

# b = torch.rand(32)
# print(b.shape)
# f = torch.rand(4, 32, 14, 14)
# b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
# print(b.shape)


# b = torch.rand(1, 32, 1, 1)
# print(b.shape)
# print(b.squeeze().shape)
# print(b.squeeze(0).shape)
# print(b.squeeze(-1).shape)
# print(b.squeeze(1).shape)
# print(b.squeeze(-4).shape)


# a = torch.rand(4, 32, 14, 14)
# b = torch.rand(1, 32, 1, 1)
#
# print(b.expand(4, 32, 14, 14).shape)
# print(b.expand(-1, 32, -1, -1).shape)
# print(b.expand(1, 32, -1, -4).shape)


# b = torch.rand(1, 32, 1, 1)
# print(b.repeat(4, 32, 1, 1).shape)
# print(b.repeat(4, 1, 1, 1).shape)
# print(b.repeat(4, 1, 32, 32).shape)

# a = torch.randn(3, 4)
# print(a)
# print(a.t())

a = torch.randn(4, 3, 32, 32)
# ak = a.transpose(1, 3).view(4, 3*32*32).view(4, 3, 32, 32)
# print(ak)

a1 = a.transpose(1, 3).contiguous().view(4, 3*32*32).view(4, 3, 32, 32)  # 这种顺序变化，改变了数据，要注意避免
print(a1.shape)
a2 = a.transpose(1, 3).contiguous().view(4, 3*32*32).view(4, 32, 32, 3).transpose(1, 3)  # 这种顺序变换，保持数据不变。
print(a2.shape)

print(torch.all(torch.eq(a, a1)))
print(torch.all(torch.eq(a, a2)))

"""
注意：【b, h, w, c】 是numpy存储图片的格式，需要这一步才能导出numpy
"""
