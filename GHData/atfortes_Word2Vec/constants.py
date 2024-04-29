import torch


CORPUS_FILENAME = 'text8'
TEST_FILE_PATH = 'wordsim353/combined.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
