# encoding=utf-8

import argparse
from pytorch_demos.demo1.main import main as pt_m1
from pytorch_demos.demo2.entry import main as pt_m2
from pytorch_demos.demo3.entry import main as pt_m3
from pytorch_demos.demo4.entry import main as pt_m4
from pytorch_demos.demo5.entry import main as pt_m5
from pytorch_demos.demo6.entry import main as pt_m6
from pytorch_demos.demo7.entry import main as pt_m7
from pytorch_demos.demo8.entry import main as pt_m8
from pytorch_demos.demo9.entry import main as pt_m9
from pytorch_demos.demo10.entry import main as pt_m10
from datetime import datetime

tasks = {
    'Pytorch Mnist': pt_m1,
    'Dcgan': pt_m2,
    'Vae': pt_m3,
    'Mnist hogwild': pt_m4,
    'Regression': pt_m5,
    'LSTM wave': pt_m6,
    'STM Name classify': pt_m7,
    'LSTM Name generating': pt_m8,
    'Word languagle model': pt_m9,
    'My CNN': pt_m10
}


def launch(func_name, dataset, epoch):
    func = tasks[func_name]
    start = datetime.now()
    print('{:-^60s}'.format(func_name))
    print('%s starts executing at %s' % (func_name.ljust(20), start.strftime('%Y/%m/%d, %H:%M:%S')))
    func(dataset, epoch)
    elapsed = datetime.now() - start
    print('%s finished, Elapsed time:%.4f s' % (func_name.ljust(20), elapsed.seconds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--all', help='run all demos', action='store_true')
    parser.add_argument('-n', '--name', help='the name of the task you want to run', type=str, default='Pytorch Mnist')
    parser.add_argument('-e', '--epoch', help='set num of epochs', type=int, default=10)
    parser.add_argument('data', type=str, help='path to data set')
    options = parser.parse_args()
    print(options)
    if options.all:
        print('Run all demos')
        for task in tasks:
            launch(task, options.data, options.epoch)
    else:
        print('Run %s' % options.name)
        launch(options.name, options.data, options.epoch)
