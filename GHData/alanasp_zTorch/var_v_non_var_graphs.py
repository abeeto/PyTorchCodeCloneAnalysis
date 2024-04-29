import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

plt_color_codes = 'bgrcmykw'

std = [0.01, 0.1, 0.3]
num_prof = 1000

filename = 'saved_exp_data/varied_v_non_varied_{}_{}_{}_{}'.format(int(100*std[0]), int(100*std[1]),
                                                                   int(100*std[2]), num_prof)


with open(filename, 'r') as file:
    file.readline()
    file.readline()
    time = np.array(list(map(float, file.readline().split(' '))))
    time = time/time[-1]*100
    aff_groups_non_var = np.array(list(map(float, file.readline().split(' '))))
    aff_groups_var = np.array(list(map(float, file.readline().split(' '))))

#fit line on aff_groups_non_var vs aff_groups_var
coeffs = np.polyfit(aff_groups_non_var, aff_groups_var, 1)
line = np.poly1d(coeffs)
plt_non_var_line, = plt.plot(time, aff_groups_non_var, label='Non-varied monitoring frequency, std={}'.format(std))
plt_var_line, = plt.plot(time, aff_groups_var, label='Varied monitoring frequency, std={}'.format(std))
plt.legend(handles=[plt_non_var_line, plt_var_line])
plt.xlabel('Simulation Time')
plt.ylabel('Number of VNF Affinity Groups (N)')
plt.show()
plt_fit_line, = plt.plot(aff_groups_non_var, line(aff_groups_non_var), color='g',
                         label='Fitted line, slope={:.2f}'.format(coeffs[0]))
plt_points = plt.scatter(aff_groups_non_var, aff_groups_var, color='b', label='Observed values'.format(coeffs[0]))
plt.legend(handles=[plt_fit_line, plt_points])
plt.xlabel('Non-varied frequency affinity groups')
plt.ylabel('Varied frequency affinity groups')
plt.show()
