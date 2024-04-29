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