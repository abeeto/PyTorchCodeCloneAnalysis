# %%
from utils import *
from datapreprocessing import processing_data

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


#%%
X_df, y_df = processing_data("is_canceled")
X_np, y_np = X_df.to_numpy(), y_df.to_numpy()
print(f"X_np's shape: {X_np.shape}")
print(f"y_np's shape: {y_np.shape}")

train_loader, val_loader, test_loader = LoadData(
    X_y=(X_np, y_np), X_y_dtype=("float", "float")
).get_dataloader([0.7, 0.2, 0.1], batch_size=64)


# %% start from here!
if __name__ == "__main__":
    # setting
    model = BinaryClassificationModel(X_np.shape[1])
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    modelwrapper = ModelWrapper(model, loss_func, optimizer)

    # training
    model = modelwrapper.train(train_loader, val_loader, max_epochs=50)

    # evaluate the model
    modelwrapper.classification_report(
        test_loader, ["Not Cancel", "Cancel"], visualize=True, binary=True
    )
