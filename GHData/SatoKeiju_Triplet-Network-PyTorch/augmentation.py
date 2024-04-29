import os
import Augmentor


def data_augmentation(number):
    root_path = './data'
    figure_list = os.listdir(root_path)

    if '.DS_Store' in figure_list:
        figure_list.remove('.DS_Store')

    for figure in figure_list:
        dir_name = 'data/' + figure + '/output'
        num_datas = len(os.listdir(dir_name))

        if os.path.exists(dir_name) and num_datas > number:
            print(f'There are already more than {number} datas in {dir_name}.')
            continue

        p = Augmentor.Pipeline('data/' + figure)

        p.crop_by_size(probability=0.5, width=40, height=40, centre=False)
        p.flip_left_right(probability=0.5)
        p.flip_top_bottom(probability=0.5)
        # p.random_distortion(probability=0.5, grid_width=6, grid_height=6, magnitude=8)
        p.rotate(probability=0.5, max_left_rotation=15, max_right_rotation=15)
        p.shear(probability=0.5, max_shear_left=20, max_shear_right=20)
        p.skew(probability=0.5, magnitude=0.3)

        p.resize(probability=1.0, width=64, height=64)

        p.sample(number - num_datas)


data_augmentation(500)
