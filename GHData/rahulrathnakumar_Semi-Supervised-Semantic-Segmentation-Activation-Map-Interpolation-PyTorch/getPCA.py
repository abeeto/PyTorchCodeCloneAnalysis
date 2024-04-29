import numpy as np
from sklearn.decomposition import PCA


def normalize(x):

    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))


def getPCA(feature, n_components = 3):
    feature = feature.detach().cpu().numpy()
    pca = PCA(n_components = n_components, whiten = True)
    N,c,h,w = feature.shape
    feature = feature.reshape(N,c, h*w)
    pcaBatch = np.zeros((N, h, w, n_components))
    for i, f in enumerate(feature):
        # PCAfeature = np.ascontiguousarray(pca.fit_transform(f))
        f_centered = f - f.mean(axis = 0)
        f_centered_local = f_centered - f_centered.mean(axis = 1).reshape(c,-1)
        pcaFit = pca.fit(f_centered_local) 
        PCAfeature = pcaFit.components_.reshape(3,h,w)
        PCAfeature = PCAfeature.transpose(1,2,0)
        PCAfeature = normalize(PCAfeature)
        pcaBatch[i] = PCAfeature
    return pcaBatch
