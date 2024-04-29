from keras.layers import Embedding, Dense, LSTM, Flatten, Conv2D, MaxPooling2D, Reshape, Conv1D, GlobalMaxPooling1D, Dropout, Concatenate, Input
from keras.models import Sequential, load_model, Model
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

print("train_x.shape: ", train_x.shape)
print("test_x.shape: ", test_x.shape)

#-------HyperParameter-----
vocab_size = len(lookupTable.keys())
maxLen = train_x.shape[1]
embedding_dim = 100
dropout_prob = (0.5, 0.8)
num_filters = 128

print("vocab_size: ", vocab_size, "maxLen: ", maxLen)

model_input = Input(shape = (maxLen,))
z = Embedding(vocab_size, embedding_dim, input_length = maxLen, name="embedding")(model_input)
z = Dropout(dropout_prob[0])(z)
print(z)

conv_blocks = []

for sz in [3, 4, 5]:
    conv = Conv1D(filters = num_filters,
                         kernel_size = sz,
                         padding = "valid",
                         activation = "relu",
                         strides = 1)(z)
    print(conv.shape)
    conv = GlobalMaxPooling1D()(conv)
    print(conv.shape)
    # conv = Flatten()(conv)
    conv_blocks.append(conv)
    
print(conv_blocks)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
z = Dropout(dropout_prob[1])(z)
z = Dense(128, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('CNN_1D_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

history = model.fit(train_x, train_y, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)

loaded_model = load_model("CNN_1D_model.h5")
print("\ntest accuracy : %.4f" % (loaded_model.evaluate(test_x, test_y)[1]))
