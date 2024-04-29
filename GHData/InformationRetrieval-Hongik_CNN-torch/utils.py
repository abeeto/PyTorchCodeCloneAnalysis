from keras.layers import Embedding, Dense, LSTM, Flatten, Conv2D, MaxPooling2D, Reshape
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from DataSet import loadDocs

# ==================================================================================================
# ========================= generate simple test model and test data set. ==========================
# ==================================================================================================
train_x = loadDocs("./Data/train_x.pkl")
train_y = loadDocs("./Data/train_y.pkl")
test_x = loadDocs("./Data/test_x.pkl")
test_y = loadDocs("./Data/test_y.pkl")
lookupTable = loadDocs("./Data/lookupTable.pkl")

model = Sequential()
vocab_size = len(lookupTable.keys())
embedding_dim = 128
maxLen = train_x.shape[1]
print(train_x.shape)
print(vocab_size, maxLen)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=maxLen))
print(model.output_shape)
# model.add(Dropout(0.3))
model.add(Reshape((embedding_dim, maxLen, -1)))
print(model.output_shape)
model.add(Conv2D(50, (5, 80), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 1)))
print(model.output_shape)

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# # earlyStop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)
# modelCheck = ModelCheckpoint("1D_cnn_best_model.h5", monitor="val_acc", mode="max", verbose=1, save_best_only=True)

# model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
# history = model.fit(train_x, train_y, epochs=15, callbacks=[modelCheck], batch_size=60, validation_split=0.2)

# loaded_model = load_model("1D_cnn_best_model.h5")
# print("\ntest accuracy : %.4f" % (loaded_model.evaluate(test_x, test_y)[1]))

import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

def summary(model, input_size, batch_size=-1, device="cuda"):
    """
    Original Codes in torchsummary package.
    That package use torch.FloatTensor, but it does not match out model(match only torch.LongTensor)
    So we scratch source codes from torchsummary, and correct it appropriately.
    """

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in ["cuda", "cpu",], "Input device is not valid, please specify 'cuda' or 'cpu'"

    """
    we corrected this lines(159-163)
    """
    if device == "cuda" and torch.cuda.is_available():

        dtype = torch.cuda.LongTensor
    else:
        dtype = torch.LongTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(layer, str(summary[layer]["output_shape"]), "{0:,}".format(summary[layer]["nb_params"]),)
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4.0 / (1024 ** 2.0))
    total_output_size = abs(2.0 * total_output * 4.0 / (1024 ** 2.0))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4.0 / (1024 ** 2.0))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary