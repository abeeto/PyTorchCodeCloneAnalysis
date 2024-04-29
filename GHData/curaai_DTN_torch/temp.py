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
    animal_url = "https://a-z-animals.com/animals/"

    count = 0
    alphabet = ''.join(sorted("qwertyuiopasdfghjklzxcvbnm".upper()))
    for letter in ['B']:
        bs = BeautifulSoup(urlopen(url + letter + '/'), 'html.parser')
        cards = bs.find_all('div', {'class': 'picture'})
        for card in cards:
            animal = card.find('b').contents[0]
            bs = BeautifulSoup(urlopen(animal_url + animal), 'html.parser').find('div', {'class': 'content'})
            images = bs.find_all('img')
            for image in images[1:]:
                try:
                    src = image['data-src']
                except:
                    src = image['src']
                download = base_url + src
                urllib.request.urlretrieve(download, 'data/temp/%d.jpg' % count)
                count += 1
