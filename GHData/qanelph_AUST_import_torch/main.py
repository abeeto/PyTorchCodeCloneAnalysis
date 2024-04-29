import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import linear_model
#import Image
from PIL import Image
import pickle
import os
X = []
y = []
import cv2
import csv
#pth = 'fft/alg/'

# for fname in os.listdir(pth):
#     a = np.load(pth + fname).flatten()
#     r = np.array([np.real(a), np.imag(a)]).flatten()
#     X.append(r)
#     y.append(0)

# pth = 'fft/no/'
# for fname in os.listdir(pth):
#     a = np.load(pth + fname).flatten()
#     r = np.array([np.real(a), np.imag(a)]).flatten()
#     X.append(r)
#     y.append(1)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
#models = (linear_model.SGDClassifier(max_iter=1000, tol=1e-3),)
#          svm.LinearSVC(C=C, max_iter=10000),
#          svm.SVC(kernel='rbf', gamma=0.7, C=C),
#          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
with open('./svc_sgd_model_full.sav', 'rb') as model_file:
    model_full = pickle.load(model_file)
    pass
#model_split = pickle.load('./svc_sgd_model.sav')
#models =
#models = (clf.fit(X_train, y_train) for clf in models)
#scores = (clf.score(X_test, y_test) for clf in models)
#img = Image.open('./test_mixed/00_06_18.bmp')
#arr_img=np.asarray(img)#.flatten()
#arr_img = arr_img.reshape((arr_img.shape[0], arr_img.shape[1], 2))
#print(arr_img.shape)
#print(r)
#arr_img.reshape(-1, arr_img.shape[-1])

# a = np.fromfile('./test_mixed/00_06_18.bmp').flatten()
# r = np.array([np.real(a), np.imag(a)]).flatten()


# img = cv2.imread('./test_mixed/00_06_18.bmp', 0)
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# r = np.array([np.real(fshift), np.imag(fshift)])#.flatten()
# pth = './test_mixed'
# r = r.reshape(1, -1)
pth = './croped_image0/'
#print(os.listdir(pth))
#img_path = os.path.join(pth, '00_06_18.bmp')
out_list = []
for index, img_name in enumerate(sorted(os.listdir(pth))):
    print(img_name)
    img_path = os.path.join(pth, img_name)
    #print(img_path)
    crop_img_count = len(os.listdir("./out0/"))
    video_time = 298
    snapshot_time = crop_img_count//video_time
    time = snapshot_time * index
    img = cv2.imread(img_path, 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    r = np.array([np.real(fshift), np.imag(fshift)])#.flatten()

    r = r.reshape(1, -1)
    label = model_full.predict(r)
    out_list.append([img_path, label, time])
    print([img_path, label, time])
#print(list( scores ))
with open("import_torch_out.csv", "w") as outfile:
    writer = csv.writer(outfile)
    for line in out_list:
        writer.writerow(line)