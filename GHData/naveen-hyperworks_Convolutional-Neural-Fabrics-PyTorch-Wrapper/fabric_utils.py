import os
import sys
import time
import math

def training_curves(file_):
    train_loss = [line.rstrip('\n') for line in open(file_+'train_loss.txt')]
    train_loss_hist = []
    train_acc_hist = []

    for loss in train_loss:
        train_loss_hist += [float(loss.split(' ')[1])]
        train_acc_hist += [100-float(loss.split(' ')[2])]


    val_loss = [line.rstrip('\n') for line in open(file_+'val_loss.txt')]
    val_loss_hist = []
    val_acc_hist = []
    for loss in val_loss:
        val_loss_hist += [float(loss.split(' ')[1])]
        val_acc_hist += [100-float(loss.split(' ')[2])]
    
    print len(train_loss_hist),  len(val_loss_hist)

    fig, ax1 = plt.subplots()
    ax1.plot(train_loss_hist,  'C0--', alpha=0.5, label='train_loss')
    ax1.plot(val_loss_hist, 'C0-', alpha=0.5, label='val_loss', linewidth=2.0)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel('cross entropy loss')
    ax1.tick_params('y')
    ax2 = ax1.twinx()
    ax2.plot(train_acc_hist, 'C1--', alpha=0.5, label='train_error')
    ax2.plot(val_acc_hist, 'C1-', alpha=0.5, label='val_error', linewidth=2.0)
    ax2.set_ylabel('Error')
    ax2.tick_params('y')
    ax1.legend(loc='upper center')
    ax2.legend(loc='best')
    fig.tight_layout()
    plt.show()

def param_counts(net):
    count = 0
    for parameter in net.parameters():
        c = 1
        for dim in parameter.size():
            c = c*dim
        count += c
    return count

term_width = int(80)

TOTAL_BAR_LENGTH = 25.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
