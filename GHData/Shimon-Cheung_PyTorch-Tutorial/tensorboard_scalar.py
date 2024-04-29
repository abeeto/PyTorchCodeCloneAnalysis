# -*- coding: utf-8 -*-
"""
@Author ：Shimon-Cheung
@Date   ：2022/4/24 10:34
@Desc   ：tensorboard中标量数据的写入操作
"""

from torch.utils.tensorboard import SummaryWriter  # 导入tensorboard包

writer = SummaryWriter(log_dir="logs")  # 指定日志文件保存的文件目录

for i in range(100):
    # 这里是距离添加数值的使用方法，调用add_scalar方法，传入的三个值分别是 图表的title 图表的x轴 图表的y轴
    writer.add_scalar("y=x", i, i)

# 最后关闭对象
writer.close()

# tensorboard --logdir=logs 在web端查看
