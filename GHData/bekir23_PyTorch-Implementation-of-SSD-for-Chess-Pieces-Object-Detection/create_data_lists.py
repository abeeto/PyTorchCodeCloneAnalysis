from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_data_path='chess_dataset/train',
                      test_data_path='chess_dataset/test',
                      output_folder='json_outputs')
