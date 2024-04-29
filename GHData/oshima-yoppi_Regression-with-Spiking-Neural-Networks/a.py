import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


def kernel(t):
    """ 畳み込みカーネル """
    return np.exp(-t/10)
    return np.exp(-t / 20) - np.exp(-t / 10)


if __name__ == '__main__':
    # 観測時間とタイムステップ
    time = 100
    dt = 0.5

    # もともとのデジタル信号を適当に作る
    spikes = np.zeros(int(time / dt))
    spikes[50] = 1
    spikes[70] = 1
    spikes[140] = 1

    t = np.arange(0, time, dt)

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    # plt.title(r'Original Spike Train $s(t)$')
    plt.title('Spike Train')
    plt.plot(t, spikes)
    plt.xlabel('time [ms]')

    plt.subplot(1, 3, 2)
    # plt.title(r'Conv. Kernel $H(t)$')
    plt.title('Kernel Function')
    plt.plot(t, kernel(t))
    plt.xlabel('time [ms]')

    plt.subplot(1, 3, 3)
    # plt.title(r'Result: $s(t) * H(t)$')
    plt.title(r'result')
    plt.plot(t, np.convolve(spikes, kernel(t))[:int(time / dt)])
    plt.xlabel('time [ms]')

    plt.show()
# fr = 100 # Hz
# dt = 1e-3
# x = np.where(np.random.rand(1, 100) < fr*dt, 1, 0)