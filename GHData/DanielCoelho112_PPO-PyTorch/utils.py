import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw
from collections import deque
import os

def plot_learning_curve(scores, x, figure_file): 
    running_avg = np.zeros(len(scores)) # why using moving averages?? Because it is much smoother and robust to outliers.
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, 1-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('running average of previous 100 scores')
    plt.savefig(figure_file)
    
   
def save_frames_as_gif(frames, path='./', filename='gym_animation2.gif'):
    """Takes a list of frames (each frame can be generated with the `env.render()` function from OpenAI gym)
    and converts it into GIF, and saves it to the specified location.
    Code adapted from this gist: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553

    Args:
        frames (list): A list of frames generated with the env.render() function
        path (str, optional): The folder in which to save the generated GIF. Defaults to './'.
        filename (str, optional): The target filename. Defaults to 'gym_animation.gif'.
    """
    imageio.mimwrite(os.path.join(path, filename), frames, fps=60)


def label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)
    
    drawer = ImageDraw.Draw(im)
    
    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f'Episode: {episode_num+1}', fill=text_color)
    
    return im