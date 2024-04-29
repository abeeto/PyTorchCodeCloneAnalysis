import sys
import random
from random import shuffle
from tqdm import tqdm
import os.path
import urllib
import urllib.error
import urllib.request
import imghdr
from ssl import CertificateError
from socket import timeout as TimeoutError
from socket import error as SocketError
import time
import pickle
import hashlib
import itertools

random.seed(42)

min_size = 7000
# api_url = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=%s"
api_url = "http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid=%s"


def hash(content):
    m = hashlib.md5()
    m.update(content)
    hex_value = m.hexdigest()
    return hex_value

class DownloadError(Exception):
    """Base class for exceptions in this module."""
    def __init__(self, message=""):
        self.message = message

def download(url, timeout, retry, sleep=0.8):
    """Downloads a file at given URL."""
    count = 0
    while True:
        try:
            f = urllib.request.urlopen(url, timeout=timeout)
            if f is None:
                raise DownloadError('Cannot open URL' + url)
            content = f.read()
            f.close()
            break
        except (urllib.error.HTTPError, CertificateError) as e:
            if isinstance(e, urllib.error.HTTPError):
                if e.code >= 400: raise DownloadError("No Access")
            count += 1
            if count > retry:
                raise DownloadError("Retry limit")
        except (TimeoutError, SocketError, IOError) as e:
            count += 1
            if count > retry:
                raise DownloadError("Retry limit")
            time.sleep(sleep)
    return content

def get_urls(id_to_synset_dict, limit=-1): 
    urls = []
    
    print("Downloading urls from %s" % api_url)

    progress_bar = tqdm(total=len(id_to_synset_dict))
    for synset_id, i in id_to_synset_dict.items():
        url = api_url % synset_id
        content = download(url, timeout=15, retry=5, sleep=0.1)
        content = content.decode('ISO-8859-1') 
        content = content.split('\n')[:-1] # last line is blank 
        
        if limit is not None: 
            shuffle(content)
            content = content[:limit]

        content = [(i, *c.split()) for c in content]
        urls += content
        progress_bar.set_postfix({"num_urls": len(urls)})
        progress_bar.update(1)

    return urls

def find_extension(content):
    extension = imghdr.what('', content) #check if valid image
    if extension == "jpeg":
        extension = "jpg"
    if extension == None:
        raise DownloadError()
    return extension

def write_file(content, file_name):
    with open(file_name, 'wb') as f:
        f.write(content)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("""Specify number of images:
        > python download_imagenet_images.py [num_imgs]""")
        exit()

    # Id to Class information
    with open('./synset_id_to_class', 'r') as f:
        id_to_class = {}
        for l in f:
            c, i = l.strip().split()
            id_to_class[i] = c

    n_imgs = int(sys.argv[1])
    if os.path.exists('valid_urls.pkl'):
        with open('valid_urls.pkl', 'rb') as f:
            urls = pickle.load(f)
    else:
        urls = get_urls(id_to_class, limit=None)

        print(f"Storing urls in 'valid_urls.pkl'")
        with open('valid_urls.pkl', 'wb') as f:
            pickle.dump(urls, f)
 
    shuffle(urls)
    print(urls[:10])

    with open('flickr_empty.png', 'rb') as f:
        empty = f.read()

    # Hash of empty image.
    empty_hex = hash(empty)

    os.makedirs('images', exist_ok=True)

    print("Starting image download")
    progress_bar = tqdm(total=n_imgs)
    i = j = 0
    good = [True] * len(urls)
    while i < n_imgs and j < len(urls):
        try:
            cl      = urls[j][0]
            name    = urls[j][1]
            url     = urls[j][2]

            response = download(url, 2, 2)

            if hash(response) == empty_hex:
                raise DownloadError()

            ext = find_extension(response)
            file_name = os.path.join("images", cl + "_" + name + "." + ext)
            write_file(response, file_name)
            progress_bar.update(1)
            j += 1
            i += 1

        except DownloadError as d:
            # print("Download error", d)
            good[j] = False
            j += 1

        except Exception as e:
            # print("Exception", e, e.__class__.__name__)
            good[j] = False
            j += 1

    progress_bar.close()
    print(f"i: {i}, j: {j}, skipped: {j - i}")

    len_before = len(urls)
    urls = list(itertools.compress(urls, good))
    len_after = len(urls)

    print("Updating valid urls to only contain urls, that actually gave a good result")
    print(f"Went from {len_before} to {len_after} elements")
    with open('valid_urls.pkl', 'wb') as f:
        pickle.dump(urls, f)

