import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 20
EVAL_BATCH_SIZE = 10
BPTT = 35
EMBEDDING_SIZE = 200
D_HID = 200
N_LAYERS = 2
N_HEAD = 2
DROPOUT = 0.2
BEST_VAL_LOSS = float('inf')
EPOCHS = 3
LEARNING_RATE = 5.0

best_model = None