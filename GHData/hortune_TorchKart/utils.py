import numpy as np
from IPython import embed
from skimage.exposure import equalize_hist

def process_frame(frame):
    #frame = 0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2]
    #frame = frame.copy()
    #frame = equalize_hist(frame).reshape(640, 480, 1)
    #for channel in range(3):
    #    frame[:,:,channel] = equalize_hist(frame[:,:,channel])
    frame = frame.astype(np.uint8)[::2, ::2,:] #.reshape(320,240,3)
    return frame


def discrete_action(action):
    d_action = [0, 0, 0, 0]
    if action[0] > 0:
        d_action[0] = 1
    elif action[0] < 0:
        d_action[1] = 1
    elif action[2] > 0:
        d_action[2] = 1
    else:
        d_action[3] = 1
    return d_action
    
def actionTransform(action):
    continuous_action = [0, 0, 0, 0, 0]
    if action.data[0] == 0:
        continuous_action[0] = 60
        continuous_action[2] = 1
    elif action.data[0] == 1:
        continuous_action[0] = -60
        continuous_action[2] = 1
    elif action.data[0] == 2:
        continuous_action[2] = 1
    return continuous_action
