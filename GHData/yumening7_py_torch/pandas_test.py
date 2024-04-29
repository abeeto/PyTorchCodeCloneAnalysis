

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 使用whitegrid (白网格) 的背景主题风格
sns.set_style("whitegrid")

# 生成一个20行6列的随机矩阵，每一列随机后都会递增0.5
data = np.random.random(size=(20, 6)) + np.arange(6)/2

# 使用seaborn绘制盒型图
sns.boxplot(data=data)

sns.despine()

plt.show()