import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pdb


# Get inputs.
error = np.loadtxt('log_x_eval_regression_error.txt')
error_real1 = np.loadtxt('log_x_eval_regression_error_1s.txt')
error_real0 = np.loadtxt('log_x_eval_regression_error_0s.txt')

x_prop = np.loadtxt('log_x_eval_proportion1.txt')
y_prop = np.loadtxt('log_y_eval_proportion1.txt')

wmmd = np.loadtxt('log_wmmd.txt')

ae_real = np.loadtxt('log_ae_real.txt')
ae_real1 = np.loadtxt('log_ae_real1.txt')
ae_real0 = np.loadtxt('log_ae_real0.txt')
ae_gen = np.loadtxt('log_ae_gen.txt')
ae_gen1 = np.loadtxt('log_ae_gen1.txt')
ae_gen0 = np.loadtxt('log_ae_gen0.txt')

x_eval_median_thinfn_1s = np.loadtxt('log_x_eval_median_thinfn_1s.txt')
x_eval_median_thinfn_0s = np.loadtxt('log_x_eval_median_thinfn_0s.txt')
y_eval_median_thinfn_1s = np.loadtxt('log_y_eval_median_thinfn_1s.txt')
y_eval_median_thinfn_0s = np.loadtxt('log_y_eval_median_thinfn_0s.txt')

hinge_unthinned = np.loadtxt('log_hinge_unthinned.txt')
hinge_thinned = np.loadtxt('log_hinge_thinned.txt')

# Make plots.
plt.figure(figsize=(10,18))
plt.subplot(611)
plt.plot(error_real1, color='blue', alpha=0.7, label='real1')
plt.plot(error_real0, color='green', alpha=0.7, label='real0')
plt.xlabel('Iteration x100')
plt.ylabel('LogReg Error')
plt.legend()

plt.subplot(612)
plt.plot(x_prop, label='data')
plt.plot(y_prop, label='gen')
plt.xlabel('Iteration x100')
plt.ylabel('Prop C1')
plt.legend()

plt.subplot(613)
plt.plot(x_eval_median_thinfn_1s, color='blue', alpha=0.7, label='real1')
plt.plot(x_eval_median_thinfn_0s, color='green', alpha=0.7, label='real0')
plt.plot(y_eval_median_thinfn_1s, color='red', alpha=0.7, label='gen1')
plt.plot(y_eval_median_thinfn_0s, color='orange', alpha=0.7, label='gen0')
plt.xlabel('Iteration x100')
plt.ylabel('Median thinfn value')
plt.legend()

plt.subplot(614)
plt.plot(wmmd, color='black', alpha=0.7)
plt.xlabel('Iteration x100')
plt.ylabel('(Weighted) MMD')

plt.subplot(615)
plt.plot(ae_real1, color='blue', alpha=0.7, label='real1')
plt.plot(ae_real0, color='green', alpha=0.7, label='real0')
plt.plot(ae_gen1, color='red', alpha=0.7, label='gen1')
plt.plot(ae_gen0, color='orange', alpha=0.7, label='gen0')
plt.xlabel('Iteration x100')
plt.ylabel('AE_loss')
plt.legend()

plt.subplot(616)
plt.plot(hinge_unthinned, color='blue', alpha=0.7, label='unthinned')
plt.plot(hinge_thinned, color='green', alpha=0.7, label='thinned')
plt.xlabel('Iteration x100')
plt.ylabel('Hinge Loss')
plt.legend()

plt.tight_layout()
plt.savefig('results.png')

# Plot wmmd for x versus mixes, and for y versus mixes.
diagnostic = 0
if not diagnostic:
    print 'No diagnostic. Skipping this section.'
else:
    plt.figure()
    mmds_xvm = np.load('mmds_xvm.npy')
    mmds_yvm = np.load('mmds_yvm.npy')
    wmmds_xvm = np.load('wmmds_xvm.npy')
    wmmds_yvm = np.load('wmmds_yvm.npy')
    plt.subplot(211)
    plt.plot(mmds_xvm, color='blue', alpha=0.7, label='data 5050')
    plt.plot(mmds_yvm, color='green', alpha=0.7, label='gen 5050')
    plt.xticks(np.arange(len(mmds_xvm)),
        ('10', '20', '30', '40', '50', '60', '70', '80', '90'))
    plt.xlabel('Percent Target Class')
    plt.ylabel('MMD')
    plt.legend()
    plt.subplot(212)
    plt.plot(wmmds_xvm, color='blue', alpha=0.7, label='data 5050')
    plt.plot(wmmds_yvm, color='green', alpha=0.7, label='gen 5050')
    plt.xticks(np.arange(len(mmds_xvm)),
        ('10', '20', '30', '40', '50', '60', '70', '80', '90'))
    plt.xlabel('Percent Target Class')
    plt.ylabel('Weighted MMD')
    plt.legend()
    plt.suptitle('MMDs of 5050 data and generated, versus other data mixes')
    plt.tight_layout()
    plt.savefig('versus_mixes_plot.png')
    print 'Saved versus_mixes_plot.png'

