import os
import glob
import random
import shutil


class DataSeparator():

    def __init__(self, rate=0.15):
        self.rate = rate
        self.data_path = os.path.abspath('./data')

    def validate_phase(self, phase):
        if not phase in ['train', 'test']:
            raise ValueError('Input "train" or "test" as dirname.')

    def get_dir_path(self, phase):
        self.validate_phase(phase)
        return os.path.join(self.data_path, phase)

    def get_dir_list(self, phase):
        self.validate_phase(phase)
        dir_path = self.get_dir_path(phase)
        dir_list = os.listdir(dir_path)
        for dir in dir_list:
            if dir.startswith('.'):
                dir_list.remove(dir)
        return dir_list

    def copy_directories(self):
        train_dirlist = self.get_dir_list('train')
        test_dirlist = self.get_dir_list('test')
        test_path = self.get_dir_path('test')
        for class_name in train_dirlist:
            if not class_name in test_dirlist:
                dir_name = os.path.join(test_path, class_name)
                os.mkdir(dir_name)

    def move_to_test(self):
        self.copy_directories()
        train_path = self.get_dir_path('train')
        test_path = self.get_dir_path('test')
        class_list = self.get_dir_list('train')
        for class_name in class_list:
            class_path = os.path.join(train_path, class_name)
            data_list = os.listdir(class_path)
            num_data = len(data_list)
            num_test = int(num_data * self.rate)
            test_idx = random.sample(range(num_data), k=num_test)
            move_to = os.path.join(test_path, class_name)
            for idx in test_idx:
                chosen_data_path = os.path.join(class_path, data_list[idx])
                shutil.move(chosen_data_path, move_to)


if __name__ == '__main__':
    d = DataSeparator()
    d.move_to_test()
