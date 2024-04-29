from utils.voc_utils import create_data_lists

if __name__ == "__main__":
    create_data_lists(
        voc07_path="/home/kmh/Documents/datasets/pascal/VOCdevkit/VOC2007",
        voc12_path="/home/kmh/Documents/datasets/pascal/VOCdevkit/VOC2012",
        output_folder="/home/kmh/Documents/datasets/pascal",
    )
