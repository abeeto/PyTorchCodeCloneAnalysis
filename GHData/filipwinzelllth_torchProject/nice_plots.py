import matplotlib.pyplot as plt
import numpy as np

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber, CB91_Purple, CB91_Violet]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

plt.figure(1)
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
dsc = np.array([0.80474, 0.80756, 0.81749, 0.82367, 0.82382, 0.81903, 0.82435, 0.82568])
dsc_e = np.array([0.00691, 0.00301, 0.00544, 0.00584, 0.00460, 0.00347, 0.00582, 0.00338])
plt.errorbar(np.arange(1, len(dsc)+1), dsc, dsc_e, linestyle='-', marker='o', markerfacecolor='white', capsize=3 )
plt.xlabel('Augmentation ->')
plt.ylabel('Dice Score')

plt.figure(2)
iou = np.array([0.68363, 0.68692, 0.69982, 0.70883, 0.70818, 0.70085, 0.70854, 0.71084])
iou_e = np.array([0.00867, 0.00374, 0.00832, 0.00851, 0.00688, 0.00470, 0.00845, 0.00467])
plt.errorbar(np.arange(1, len(iou)+1), iou, iou_e, color=CB91_Pink, linestyle='-', marker='o', markerfacecolor='white', capsize=3 )
plt.xlabel('Augmentation ->')
plt.ylabel('IoU')


plt.show()
