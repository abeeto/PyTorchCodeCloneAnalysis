from matplotlib import pyplot as plt


def show(title, data, save=False):
    plt.title(title)
    for key in data.keys():
        x = [i for i in range(len(data[key]))]
        plt.plot(x, data[key], title=key)
    plt.legend(list(data.keys()))
    if save:
        plt.savefig('./{}.png'.format(title))
    plt.show()


def models_plt(data, name='result'):
    plt.figure(figsize=(16, 16))
    start = 421
    for k in data:
        plt.subplot(start)
        plt.title(k + ' acc')
        plt.plot([i for i in range(len(data[k]['train_acc']))], data[k]['train_acc'], label='train')
        plt.plot([i for i in range(len(data[k]['val_acc']))], data[k]['val_acc'], label='test')
        plt.legend(['train', 'test'])
        start += 1
        plt.subplot(start)
        plt.title(k + ' loss')
        plt.plot([i for i in range(len(data[k]['train_loss']))], data[k]['train_loss'], label='train')
        plt.plot([i for i in range(len(data[k]['val_loss']))], data[k]['val_loss'], label='test')
        plt.legend(['train', 'test'])
        start += 1
    plt.savefig('./{}_models.png'.format(name), dpi=200)
    plt.show()


def models_all(data, name='result'):
    all_test_loss = {'base': data['base_result']['val_loss'], 'avg': data['avg_result']['val_loss'],
                     'reduce': data['reduce_result']['val_loss'],
                     'increase': data['increase_result']['val_loss']}
    all_test_acc = {'base': data['base_result']['val_acc'], 'avg': data['avg_result']['val_acc'],
                    'reduce': data['reduce_result']['val_acc'],
                    'increase': data['increase_result']['val_acc']}
    all_train_acc = {'base': data['base_result']['train_acc'], 'avg': data['avg_result']['train_acc'],
                     'reduce': data['reduce_result']['train_acc'],
                     'increase': data['increase_result']['train_acc']}
    all_train_loss = {'base': data['base_result']['train_loss'], 'avg': data['avg_result']['train_loss'],
                      'reduce': data['reduce_result']['train_loss'],
                      'increase': data['increase_result']['train_loss']}
    plt.figure(figsize=(16, 16))
    plt.subplot(321)
    plt.title('train acc')
    all__ = all_train_acc
    for k in all__:
        s = all__[k]
        plt.plot([i for i in range(1, len(s) + 1)], s, label=k)
    plt.legend(list(all__.keys()))

    plt.subplot(322)
    plt.title('train loss')
    all__ = all_train_loss
    for k in all__:
        s = all__[k]
        plt.plot([i for i in range(1, len(s) + 1)], s, label=k)
    plt.legend(list(all__.keys()))
    plt.subplot(323)
    plt.title('test acc')
    all__ = all_test_acc
    for k in all__:
        s = all__[k]
        plt.plot([i for i in range(1, len(s) + 1)], s, label=k)
    plt.legend(list(all__.keys()))
    plt.subplot(324)
    plt.title('test loss')
    all__ = all_test_loss
    for k in all__:
        s = all__[k]
        plt.plot([i for i in range(1, len(s) + 1)], s, label=k)
    plt.legend(list(all__.keys()))
    plt.subplot(325)
    plt.title('abs acc ')
    all__ = all_test_acc
    all1 = all_train_acc
    for k in all__:
        s = all__[k]
        s1 = all1[k]
        plt.plot([i for i in range(1, len(s) + 1)], [(i - j) for i, j in zip(s, s1)], label=k)
    plt.legend(list(all__.keys()))
    plt.subplot(326)
    plt.title('abs loss ')
    all__ = all_test_loss
    all1 = all_train_loss
    for k in all__:
        s = all__[k]
        s1 = all1[k]
        plt.plot([i for i in range(1, len(s) + 1)], [(i - j) for i, j in zip(s, s1)], label=k)
    plt.legend(list(all__.keys()))
    plt.savefig('./{}_all.png'.format(name), dpi=200)
    plt.show()
