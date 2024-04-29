import torch

t1 = torch.arange(4)
# 加法中的广播
print(t1 + t1)
print(t1 + 1)

# ---------相同维度，不同形状的张量之间的计算

t2 = torch.tensor([1, 2, 3, 4])
t3 = torch.full([3, 4], fill_value=0, dtype=torch.int16)
# t2 发生了广播，复制了3次和t3形状相同后相加
print(t2 + t3)

# 列广播 相加
# 3行1列全1矩阵
t4 = torch.ones(3, 1)
print(t3 + t4)

# 三维张量的广播

t5 = torch.zeros(3, 4, 5)
t6 = torch.ones(3, 4, 1)
print(t5 + t6)

# 广播过程就是复制那个为1尺寸的对象。


# 不同维度张量相加，先转换成相同维度
# 举例升维
t7 = torch.arange(4).reshape(2, 2)
t7 = t7.reshape(1, 2, 2)

t7 = t7.reshape(1, 1, 2, 2)

t8 = torch.zeros(3, 2, 2)
print(t3 + t2)

# -----------------逐点运算
'''
    torch.add 加法
    .subtract 减法
    .multiply 乘法
    .divide 除法
    .abs 绝对值
    .ceil 向上取整
    .floor 向下取整
    .round 四舍五入取整
    .neg 返回相反的数，负号
    
    若要对原对象本身进行修改，则考虑使用方法名+下划线=“方法_()”的表达形式。
    对对象本身进行修改。
    举例：abs_()
    注意：expm1和log1p,视频第50分钟
'''

# 排序运算 torch.sort， dim参数可以用来选择

# 闵式距离 torch.dict
'''
    dist函数可计算闵式距离（闵可夫斯基距离），通过输入不同的p值，可以计算
    多种类型的距离，如欧式距离、街道距离等。闵可夫斯基距离公式如下：
        D(x, y) = (sigma(u=1~n)|xu-yu|^p)^(1/p)
    p为1是街道距离（曼哈顿距离），p为2时为欧氏距离。
'''

# 比较运算
'''
 torch.eq 比较两个张量是否相等
 == 也可以用来判断
 .equal 判断两个张量是否相同
 .gt 判断t1是否大于t2
 .lt 小于
 .ge 大于等于
 .le 小于等于
 .ne 不等于
'''
