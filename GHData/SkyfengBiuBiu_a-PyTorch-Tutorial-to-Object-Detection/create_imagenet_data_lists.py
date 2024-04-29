from utils import create_data_lists_imagenet

if __name__ == '__main__':
    create_data_lists_imagenet(imagenet_path="/home/fengy/Downloads/ILSVRC2015_VID/ILSVRC2015",
                      output_folder='./imagenet_dataset')
