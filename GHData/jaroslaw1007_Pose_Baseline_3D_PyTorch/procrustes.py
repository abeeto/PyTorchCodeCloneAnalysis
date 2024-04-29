import numpy as np

def compute_similarity_transform(x, y, complete_optimal_scale=False):
    mu_x = x.mean(0)
    mu_y = y.mean(0)

    x0 = x - mu_x
    y0 = y - mu_y

    ss_x = (x0 ** 2.).sum()
    ss_y = (y0 ** 2.).sum()

    # centered forbenius norm
    norm_x = np.sqrt(ss_x)
    norm_y = np.sqrt(ss_y)

    # scale to equal (unit) norm
    x0 = x0 / norm_x
    y0 = y0 / norm_y

    # optimum rotation matrix of y
    # A -> Affinity matrix
    A = np.dot(x0.T, y0)
    u, s, v = np.linalg.svd(A, full_matrices=False)
    v = v.T
    t = np.dot(v, u.T)

    det_t = np.linalg.det(t)
    v[:, -1] *= np.sign(det_t) 
    s[-1] *= np.sign(det_t)
    t = np.dot(v, u.T)

    trace_ta = s.sum()

    if complete_optimal_scale:
        b = trace_ta * norm_x / norm_y
        d = 1 - trace_ta ** 2
        z = norm_x * trace_ta * np.dot(y0, t) + mu_x
    else:
        b = 1
        d = 1 + ss_y / ss_x - 2 * trace_ta * norm_y / norm_x 
        z = norm_y * np.dot(y0, t) + mu_x

    c = mu_x - b * np.dot(mu_y, t)

    return d, z, t, b, c

