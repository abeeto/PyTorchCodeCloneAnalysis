try:
    import torchvision
    import torch.nn as nn
    import torch
    import torch.nn.functional as F
    from torchvision import transforms,models, datasets
    from torchsummary import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = models.vgg16( pretrained = True ).to( device )
    summary(
        model,
        torch.zeros(
            1,
            3,
            224,
            224
        )
    )

    print( model )


    import torch
    import torch.nn.functional as F
    from torchvision import transforms, models, datasets
    import matplotlib.pyplot as plt
    from PIL import Image
    from torch import optim
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    import cv2, glob, numpy as np, pandas as pd
    from glob import glob
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset

    # run this command in the terminal
    # !pip install kaggle
    from google.colab import files

except Exception:
    print( "First part of the section on transfer learning for image classification")


# pip install -qU face-alignment
import face_alignment, cv2

fa_type_FaceAlignment = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._2D,
    flip_input = False,
    device = 'cpu'
)

input_type_ndarray = cv2.imread('data/Hema.JPG')
preds_type_ndarray = fa_type_FaceAlignment.get_landmarks(input_type_ndarray)[0]
print(preds_type_ndarray.shape)

import matplotlib.pyplot as plt

fig_type_Figure, ax_type_AxesSubPlot = plt.subplots(figsize=(5, 5))

plt.imshow(
    cv2.cvtColor(
        cv2.imread('data/Hema.JPG'),
        cv2.COLOR_BGR2RGB
    )
)

ax_type_AxesSubPlot.scatter(
    preds_type_ndarray[:,0],
    preds_type_ndarray[:, 1],
    marker = '+',
    c = 'r'
)

# test for where and values of x and y do from numpy ndarray perspective on the image
# ax_type_AxesSubPlot.scatter(
#     10,
#     20
# )

plt.show()



fa_type_FaceAlignment = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._3D,
    flip_input = False,
    device = 'cpu'
)

input_type_ndarray = cv2.imread('data/Hema.JPG')
preds_type_ndarray = fa_type_FaceAlignment.get_landmarks(input_type_ndarray)[0]
import pandas as pd
df_type_DataFrame = pd.DataFrame(preds_type_ndarray)
df_type_DataFrame.columns = ['x', 'y', 'z']
import plotly.express as px
fig_type_Figure = px.scatter_3d(
    df_type_DataFrame,
    x = 'x',
    y = 'y',
    z = 'z'
)
fig_type_Figure.show()
# open a webpage 3d plot that is interactive showing the points on the face but in 3d