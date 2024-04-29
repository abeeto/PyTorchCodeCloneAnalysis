import numpy as np
from scipy.misc import imsave, imread, imresize
from urllib.request import urlopen
from bs4 import BeautifulSoup
import urllib.request

# input image shape should be (?, ?, 4)
def rgba2rgb(input_path, output_path):
    rgba = imread(input_path)
    shape = rgba.shape

    alpha = rgba[:, :, 3].reshape(shape[0], shape[1], 1) / 255
    r_alpha = 1 - alpha

    image = rgba[:, :, :3]
    white = np.ones((shape[0], shape[1], 3)) * 255

    rgb = np.multiply(image, alpha) + np.multiply(white, r_alpha)
    imsave(output_path, rgb)


if __name__ == '__main__':
    base_url = "https://a-z-animals.com"
    url = "https://a-z-animals.com/animals/pictures/"

    count = 0
    alphabet = ''.join(sorted("qwertyuiopasdfghjklzxcvbnm".upper()))
    for letter in alphabet:
        bs = BeautifulSoup(urlopen(url + letter + '/'))
        a = bs.find_all('img')
        for x in a:
            if 'media' in x['src']:
                download_url = base_url + x['src']
                urllib.request.urlretrieve(download_url, 'data/animal/%d.jpg' % count)
                count += 1
