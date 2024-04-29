#%%
from utils import *
from datapreprocessing import processing_data

from sklearn.model_selection import train_test_split

# start from here!
if __name__ == "__main__":
    # test classifiers
    X_df, y_df = processing_data("is_canceled")
    X_np, y_np = X_df.to_numpy(), y_df.to_numpy()
    mlmodelwrapper = MLModelWrapper(X_np, y_np)
    mlmodelwrapper.quick_test("classifier")

    # test regressors
    X_df, y_df = processing_data("adr")
    X_np, y_np = X_df.to_numpy(), y_df.to_numpy()
    mlmodelwrapper = MLModelWrapper(X_np, y_np)
    mlmodelwrapper.quick_test("regressor")
