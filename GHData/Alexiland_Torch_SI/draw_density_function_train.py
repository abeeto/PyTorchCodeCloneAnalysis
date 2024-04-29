import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import numpy as np
from scipy.optimize import curve_fit
import scipy

lw = 2

font_sup = 18
font_big = 14
font_mid = 14
font_small = 12

def maxwell_boltzmann_cdf(x,a):
    term_1 = scipy.special.erf(x/np.sqrt(2)/a)
    term_2 = np.sqrt(2/np.pi)*x*np.exp(-np.power(x,2)/(2*a*a))/a
    return term_1 - term_2

def guassian_cdf(x,u,sigma):
    term_1 = 1
    term_2 = scipy.special.erf((x-u)/np.sqrt(2)/sigma)
    return (term_1 + term_2)/2
    
def least_squares(x,y):
    x_ = x.mean()
    y_ = y.mean()
    m = np.zeros(1)
    n = np.zeros(1)
    k = np.zeros(1)
    p = np.zeros(1)
    for i in np.arange(len(x)):
        k = (x[i]-x_)* (y[i]-y_)
        m += k
        p = np.square( x[i]-x_ )
        n = n + p
    a = m/n
    b = y_ - a* x_
    return a,b

def formatnum(x, pos):
    return "%.1f" % (x*100)+"%"
formatter = FuncFormatter(formatnum)

def add_mark(x,y,plt,c="k"):
    plt.plot([x,],[y,], 'k*', lw = lw+2)
    plt.plot([x,x], [y,0], linewidth=lw/1.5, color=c, linestyle='--')
    plt.plot([x,0], [y,y], linewidth=lw/1.5, color=c, linestyle='--')
    

fig = plt.figure()    

csv_path = "./"
image_complexity_list = []
for i in range(10):
    full_csv_path = csv_path + "csv_file/seg_citys_592_"+str(i)+".train_preprocessing.SI.train.csv"
    csv_file = open(full_csv_path, "r")
    lines = csv_file.readlines()
    # drop the first line
    lines = lines[1:len(lines)]
    for line in lines:
        image_complexity = float(line.split(',')[2])
        image_complexity_list.append(image_complexity)
image_complexity_list.sort()
y_ax = [i/float(len(image_complexity_list)) for i in range(len(image_complexity_list))]

# linear
a, b = least_squares(np.array(image_complexity_list), np.array(y_ax))
a, b = np.squeeze(a), np.squeeze(b)
y_line = [0.0, 1.0]
x_line = [(y-b)/a for y in y_line]

print("max_x: ", x_line[1])
print("min_x: ", x_line[0])
# save the max and min x to log
print("# Samples in Citys: ",len(image_complexity_list))
np.save("dataset_distribution/citys_592_train_linear.npy", np.array(x_line))

popt, pcov = curve_fit(maxwell_boltzmann_cdf, np.array(image_complexity_list), np.array(y_ax))
logistic_y = maxwell_boltzmann_cdf(np.array(image_complexity_list),popt[0])
print("a in S-curve: ", popt)
print("====================")

np.save("dataset_distribution/citys_592_train_S.npy", np.array(popt))

np.save("dataset_distribution/citys_592_train_ori_IC.npy", np.array(image_complexity_list))
np.save("dataset_distribution/citys_592_train_ori_prob.npy", np.array(y_ax))

plt.plot(image_complexity_list, y_ax, 'r', lw=lw+1, label=r"$\bf{CityScapes}$")
plt.plot(image_complexity_list, logistic_y, 'r:', lw=lw)

ax=plt.gca()



image_complexity_list = []
for i in range(10):
    full_csv_path = csv_path + "csv_file/seg_bdd_400_"+str(i)+".train_preprocessing.SI.train.csv"
    csv_file = open(full_csv_path, "r")
    lines = csv_file.readlines()
    # drop the first line
    lines = lines[1:len(lines)]
    for line in lines:
        image_complexity = float(line.split(',')[2])
        image_complexity_list.append(image_complexity)
image_complexity_list.sort()
y_ax = [i/float(len(image_complexity_list)) for i in range(len(image_complexity_list))]

a, b = least_squares(np.array(image_complexity_list), np.array(y_ax))
a, b = np.squeeze(a), np.squeeze(b)
y_line = [0.0, 1.0]
x_line = [(y-b)/a for y in y_line]

print("max_x: ", x_line[1])
print("min_x: ", x_line[0])
# save the max and min x to log
print("# Samples in BDD: ",len(image_complexity_list))
np.save("dataset_distribution/bdd_400_train_linear.npy", np.array(x_line))


popt, pcov = curve_fit(maxwell_boltzmann_cdf, np.array(image_complexity_list), np.array(y_ax))
logistic_y = maxwell_boltzmann_cdf(np.array(image_complexity_list),popt[0])
print("a in S-curve: ", popt)
print("====================")
np.save("dataset_distribution/bdd_400_train_S.npy", np.array(popt))

np.save("dataset_distribution/bdd_400_train_ori_IC.npy", np.array(image_complexity_list))
np.save("dataset_distribution/bdd_400_train_ori_prob.npy", np.array(y_ax))



plt.plot(image_complexity_list, y_ax, 'g', lw=lw+1, label=r"$\bf{BDD}$")
plt.plot(image_complexity_list, logistic_y, 'g:', lw=lw)


image_complexity_list = []
for i in range(30):
    full_csv_path = csv_path + "csv_file/seg_camvid_400_"+str(i)+".train_preprocessing.SI.train.csv"
    if os.path.exists(full_csv_path):
        csv_file = open(full_csv_path, "r")
        lines = csv_file.readlines()
        # drop the first line
        lines = lines[1:len(lines)]
        for line in lines:  
            image_complexity = float(line.split(',')[2])
            image_complexity_list.append(image_complexity)
image_complexity_list.sort()
y_ax = [i/float(len(image_complexity_list)) for i in range(len(image_complexity_list))]

a, b = least_squares(np.array(image_complexity_list), np.array(y_ax))
a, b = np.squeeze(a), np.squeeze(b)
y_line = [0.0, 1.0]
x_line = [(y-b)/a for y in y_line]

print("max_x: ", x_line[1])
print("min_x: ", x_line[0])
# save the max and min x to log
print("# Samples in Camvid: ",len(image_complexity_list))
np.save("dataset_distribution/camvid_400_train_linear.npy", np.array(x_line))


popt, pcov = curve_fit(maxwell_boltzmann_cdf, np.array(image_complexity_list), np.array(y_ax))
logistic_y = maxwell_boltzmann_cdf(np.array(image_complexity_list),popt[0])
print("a in S-curve: ", popt)
print("====================")
np.save("dataset_distribution/camvid_400_train_S.npy", np.array(popt))

np.save("dataset_distribution/camvid_400_train_ori_IC.npy", np.array(image_complexity_list))
np.save("dataset_distribution/camvid_400_train_ori_prob.npy", np.array(y_ax))


plt.plot(image_complexity_list, y_ax, 'b', lw=lw+1, label=r"$\bf{CamVid}$")
plt.plot(image_complexity_list, logistic_y, 'b:', lw=lw)


image_complexity_list = []
for i in range(30):
    full_csv_path = csv_path + "csv_file/CIFAR10_{}.train_rndCropFlip.SI.train.csv".format(i)
    if os.path.exists(full_csv_path):
        csv_file = open(full_csv_path, "r")
        lines = csv_file.readlines()
        # drop the first line
        lines = lines[1:len(lines)]
        for line in lines:
            try:
                image_complexity = float(line.split(',')[2])
                image_complexity_list.append(image_complexity)
            except:
                continue
image_complexity_list.sort()
y_ax = [i/float(len(image_complexity_list)) for i in range(len(image_complexity_list))]

a, b = least_squares(np.array(image_complexity_list), np.array(y_ax))
a, b = np.squeeze(a), np.squeeze(b)
y_line = [0.0, 1.0]
x_line = [(y-b)/a for y in y_line]

print("max_x: ", x_line[1])
print("min_x: ", x_line[0])
# save the max and min x to log
print("# Samples in CIFAR-10: ",len(image_complexity_list))
np.save("dataset_distribution/CIFAR-10_32_train_linear.npy", np.array(x_line))


popt, pcov = curve_fit(maxwell_boltzmann_cdf, np.array(image_complexity_list), np.array(y_ax))
logistic_y = maxwell_boltzmann_cdf(np.array(image_complexity_list),popt[0])
print("a in S-curve: ", popt)
print("====================")
np.save("dataset_distribution/CIFAR-10_32_train_S.npy", np.array(popt))

np.save("dataset_distribution/CIFAR-10_32_train_ori_IC.npy", np.array(image_complexity_list))
np.save("dataset_distribution/CIFAR-10_32_train_ori_prob.npy", np.array(y_ax))


plt.plot(image_complexity_list, y_ax, 'c', lw=lw+1, label=r"$\bf{CIFAR-10}$")
plt.plot(image_complexity_list, logistic_y, 'c:', lw=lw)


image_complexity_list = []
for i in range(30):
    full_csv_path = csv_path + "csv_file/CIFAR100_{}.train_rndCropFlip.SI.train.csv".format(i)
    if os.path.exists(full_csv_path):
        csv_file = open(full_csv_path, "r")
        lines = csv_file.readlines()
        # drop the first line
        lines = lines[1:len(lines)]
        for line in lines:  
            try:
                image_complexity = float(line.split(',')[2])
                image_complexity_list.append(image_complexity)
            except:
                continue
image_complexity_list.sort()
y_ax = [i/float(len(image_complexity_list)) for i in range(len(image_complexity_list))]

a, b = least_squares(np.array(image_complexity_list), np.array(y_ax))
a, b = np.squeeze(a), np.squeeze(b)
y_line = [0.0, 1.0]
x_line = [(y-b)/a for y in y_line]

print("max_x: ", x_line[1])
print("min_x: ", x_line[0])
# save the max and min x to log
print("# Samples in CIFAR-100: ",len(image_complexity_list))
np.save("dataset_distribution/CIFAR-100_32_train_linear.npy", np.array(x_line))


popt, pcov = curve_fit(maxwell_boltzmann_cdf, np.array(image_complexity_list), np.array(y_ax))
logistic_y = maxwell_boltzmann_cdf(np.array(image_complexity_list),popt[0])
print("a in S-curve: ", popt)
print("====================")
np.save("dataset_distribution/CIFAR-100_32_train_S.npy", np.array(popt))

np.save("dataset_distribution/CIFAR-100_32_train_ori_IC.npy", np.array(image_complexity_list))
np.save("dataset_distribution/CIFAR-100_32_train_ori_prob.npy", np.array(y_ax))


plt.plot(image_complexity_list, y_ax, 'm', lw=lw+1, label=r"$\bf{CIFAR-100}$")
plt.plot(image_complexity_list, logistic_y, 'm:', lw=lw)



leg = ax.legend(fontsize=font_mid)
plt.xlabel("Image complexity", fontweight="bold", fontsize=font_big)
plt.ylabel("Cumulative Distribution Function", fontweight="bold", fontsize=font_big)

###################################################################################

image_complexity_list = []
for i in range(30):
    full_csv_path = csv_path + "csv_file/CIFAR10_{}.train_no_preprocessing.SI.train.csv".format(i)
    if os.path.exists(full_csv_path):
        csv_file = open(full_csv_path, "r")
        lines = csv_file.readlines()
        # drop the first line
        lines = lines[1:len(lines)]
        for line in lines:
            try:
                image_complexity = float(line.split(',')[2])
                image_complexity_list.append(image_complexity)
            except:
                continue
image_complexity_list.sort()
y_ax = [i/float(len(image_complexity_list)) for i in range(len(image_complexity_list))]

a, b = least_squares(np.array(image_complexity_list), np.array(y_ax))
a, b = np.squeeze(a), np.squeeze(b)
y_line = [0.0, 1.0]
x_line = [(y-b)/a for y in y_line]

print("max_x: ", x_line[1])
print("min_x: ", x_line[0])
# save the max and min x to log
print("# Samples in CIFAR-10: ",len(image_complexity_list))
np.save("dataset_distribution/CIFAR-10_32_train_no_pre_linear.npy", np.array(x_line))


popt, pcov = curve_fit(maxwell_boltzmann_cdf, np.array(image_complexity_list), np.array(y_ax))
logistic_y = maxwell_boltzmann_cdf(np.array(image_complexity_list),popt[0])
print("a in S-curve: ", popt)
print("====================")
np.save("dataset_distribution/CIFAR-10_32_train_no_pre_S.npy", np.array(popt))

np.save("dataset_distribution/CIFAR-10_32_train_no_pre_ori_IC.npy", np.array(image_complexity_list))
np.save("dataset_distribution/CIFAR-10_32_train_no_pre_ori_prob.npy", np.array(y_ax))


plt.plot(image_complexity_list, y_ax, 'c', lw=lw+1, label=r"$\bf{CIFAR-10-without-pre}$")
plt.plot(image_complexity_list, logistic_y, 'c:', lw=lw)


image_complexity_list = []
for i in range(30):
    full_csv_path = csv_path + "csv_file/CIFAR100_{}.train_no_preprocessing.SI.train.csv".format(i)
    if os.path.exists(full_csv_path):
        csv_file = open(full_csv_path, "r")
        lines = csv_file.readlines()
        # drop the first line
        lines = lines[1:len(lines)]
        for line in lines:
            try:
                image_complexity = float(line.split(',')[2])
                image_complexity_list.append(image_complexity)
            except:
                continue
image_complexity_list.sort()
y_ax = [i/float(len(image_complexity_list)) for i in range(len(image_complexity_list))]

a, b = least_squares(np.array(image_complexity_list), np.array(y_ax))
a, b = np.squeeze(a), np.squeeze(b)
y_line = [0.0, 1.0]
x_line = [(y-b)/a for y in y_line]

print("max_x: ", x_line[1])
print("min_x: ", x_line[0])
# save the max and min x to log
print("# Samples in CIFAR-10: ",len(image_complexity_list))
np.save("dataset_distribution/CIFAR-100_32_train_no_pre_linear.npy", np.array(x_line))


popt, pcov = curve_fit(maxwell_boltzmann_cdf, np.array(image_complexity_list), np.array(y_ax))
logistic_y = maxwell_boltzmann_cdf(np.array(image_complexity_list),popt[0])
print("a in S-curve: ", popt)
print("====================")
np.save("dataset_distribution/CIFAR-100_32_train_no_pre_S.npy", np.array(popt))

np.save("dataset_distribution/CIFAR-100_32_train_no_pre_ori_IC.npy", np.array(image_complexity_list))
np.save("dataset_distribution/CIFAR-100_32_train_no_pre_ori_prob.npy", np.array(y_ax))


plt.plot(image_complexity_list, y_ax, 'c', lw=lw+1, label=r"$\bf{CIFAR-10-without-pre}$")
plt.plot(image_complexity_list, logistic_y, 'c:', lw=lw)
################################################################################
image_complexity_list = []
for i in range(30):
    full_csv_path = csv_path + "csv_file/ImageNet_{}.train_rndCropFlip.SI.train.csv".format(i)
    if os.path.exists(full_csv_path):
        csv_file = open(full_csv_path, "r")
        lines = csv_file.readlines()
        # drop the first line
        lines = lines[1:len(lines)]
        for line in lines:  
            try:
                image_complexity = float(line.split(',')[2])
                image_complexity_list.append(image_complexity)
            except:
                continue
image_complexity_list.sort()
y_ax = [i/float(len(image_complexity_list)) for i in range(len(image_complexity_list))]

a, b = least_squares(np.array(image_complexity_list), np.array(y_ax))
a, b = np.squeeze(a), np.squeeze(b)
y_line = [0.0, 1.0]
x_line = [(y-b)/a for y in y_line]

print("max_x: ", x_line[1])
print("min_x: ", x_line[0])
# save the max and min x to log
print("# Samples in ImageNet: ",len(image_complexity_list))
np.save("dataset_distribution/ImageNet_32_train_linear.npy", np.array(x_line))


popt, pcov = curve_fit(maxwell_boltzmann_cdf, np.array(image_complexity_list), np.array(y_ax))
logistic_y = maxwell_boltzmann_cdf(np.array(image_complexity_list),popt[0])
print("a in S-curve: ", popt)
print("====================")
np.save("dataset_distribution/ImageNet_32_train_S.npy", np.array(popt))

np.save("dataset_distribution/ImageNet_32_train_ori_IC.npy", np.array(image_complexity_list))
np.save("dataset_distribution/ImageNet_32_train_ori_prob.npy", np.array(y_ax))


plt.plot(image_complexity_list, y_ax, 'k', lw=lw+1, label=r"$\bf{ImageNet}$")
plt.plot(image_complexity_list, logistic_y, 'k:', lw=lw)



leg = ax.legend(fontsize=font_mid)
plt.xlabel("Image complexity", fontweight="bold", fontsize=font_big)
plt.ylabel("Cumulative Distribution Function", fontweight="bold", fontsize=font_big)


plt.xlim(xmin = 0)
plt.ylim(ymin = 0)

leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(lw)

ax.spines['bottom'].set_linewidth(lw)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(lw)
ax.spines['left'].set_color('black')
ax.spines['top'].set_linewidth(lw)
ax.spines['top'].set_color('black')
ax.spines['right'].set_linewidth(lw)
ax.spines['right'].set_color('black')
plt.tight_layout()
plt.savefig("figures/CDF_train.pdf")
