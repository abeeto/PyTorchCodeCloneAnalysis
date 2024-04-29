
import torch

a = torch.rand(2,2)
a = a * 10
print(a)

# .floor() 向下取整数
print(torch.floor(a))

# .ceil() 向上取整数
print(torch.ceil(a))

# .round() 四舍五入
print(torch.round(a))

# .trunc() 剪裁,只取整数部分
print(torch.trunc(a))

# .frac() 只取小数部分
print(torch.frac(a))

# % 取余
print(a % 2)