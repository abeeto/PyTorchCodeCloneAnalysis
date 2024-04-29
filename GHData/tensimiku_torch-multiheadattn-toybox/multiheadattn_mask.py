import torch


# set transformer enc
attn_layer = torch.nn.MultiheadAttention(8, 4, dropout=0, bias=False)

def print_hcut(*vals):
    print('---'*30)
    print(*vals)
    print('---'*30)



def print_test_mask(x, src_mask, src_key_mask):
    oval, oweight = attn_layer(x, x, x)
    mval, mweight = attn_layer(x, x, x, key_padding_mask=src_key_mask, attn_mask=src_mask)
    print('--- vals (w/o mask, w/ mask)')
    print(oval)
    print(mval)
    print('--- attn weights (w/o mask, w/ mask)')
    print(oweight)
    print(mweight)




# set some input x
# x = torch.randn(4, 2, 8) # l b c
x = torch.ones([4, 2, 8]) # l b c
x[:2, :, :4]  = 0.
x[2:, :, 4:]  = 0.


# test mask
src_mask = torch.ones([8, 4, 4]) # test (numhead*b, l, l) shape
src_mask[:4] *= -999999999 # first batch?
src_key_mask = torch.zeros([2, 4]) # test b, l shape
print_hcut('---'*10, 'test', -999999999)
print_test_mask(x, src_mask, src_key_mask)

# test another mask
src_mask = torch.ones([8, 4, 4]) # test (numhead*b, l, l) shape
src_mask[:4, :, 0] *= -torch.inf # some elements?
src_mask[:4, :, 3] *= -torch.inf # some elements?
src_mask[:4, :, 1] *= -torch.inf # some elements?

print_hcut('---'*10, 'test', '-inf')
print_test_mask(x, src_mask, src_key_mask)

# test bitwise mask
src_mask = torch.zeros([8, 4, 4], dtype=torch.bool) # test (numhead*b, l, l) shape
print_hcut('---'*10, 'test', 'bool')
print_test_mask(x, src_mask, src_key_mask)

# test bitwise mask
print_hcut('---'*10, 'test', 'bool w/ row out')
src_mask = torch.zeros([8, 4, 4], dtype=torch.bool) # test (numhead*b, l, l) shape
src_mask[:4, 0] = True
src_mask[:4, 3] = True
src_mask[4, 1] = True
print_test_mask(x, src_mask, src_key_mask) # row out => all attention goes 0 -> nan


# test bitwise mask
print_hcut('---'*10, 'test', 'bool w/ col out')
src_mask = torch.zeros([8, 4, 4], dtype=torch.bool) # test (numhead*b, l, l) shape
src_mask[:4, :, 0] = True # col out => specific attention goes 0 -> val
src_mask[:4, :, 3] = True
src_mask[4:, :, 1:] = True
print_test_mask(x, src_mask, src_key_mask)


# test key mask
src_mask = torch.zeros([8, 4, 4], dtype=torch.bool) # test (numhead*b, l, l) shape
src_key_mask = torch.zeros([2, 4], dtype=torch.bool) # test b, l shape
src_key_mask[0, :2] = True

print_hcut('---'*10, 'keymask test')
print_test_mask(x, src_mask, src_key_mask)

# test both mask for transformer layer
print_hcut('---'*10, 'srcmask + keymask test')
src_mask = torch.zeros([8, 4, 4], dtype=torch.bool) # test (numhead*b, l, l) shape
src_mask[:4, :, 2] = True
src_mask[:4, :, 3] = True
src_mask[4:, :, 1:] = True
src_key_mask = torch.zeros([2, 4], dtype=torch.bool) # test b, l shape
src_key_mask[0, [2, 3]] = True

print_test_mask(x, src_mask, src_key_mask)

# test both mask for transformer layer
src_mask = torch.zeros([8, 4, 4], dtype=torch.bool) # test (numhead*b, l, l) shape
src_mask[:4, :, 2] = True
src_mask[:4, :, 3] = True
src_mask[4:, :, 1:] = True
src_key_mask = torch.zeros([2, 4], dtype=torch.bool) # test b, l shape
src_key_mask[0, [2, 3]] = True
src_key_mask[1, 0] = True
# 2nd batch should be NaN
print_test_mask(x, src_mask, src_key_mask)
