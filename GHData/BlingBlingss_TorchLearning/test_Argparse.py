# Reference:  https://www.jianshu.com/p/00425f6c0936

import argparse
parser = argparse.ArgumentParser()
parser.description = "我在这里"
# 如果在参数前面不加--，则在命令行中输入时不用输入参数名字
parser.add_argument("--ParA", default=4, help='我是A', type=int)
parser.add_argument("--ParB", default=5, help="我是B", type=int)
args = parser.parse_args()
print('A和B的乘积是：', args.ParA*args.ParB)
