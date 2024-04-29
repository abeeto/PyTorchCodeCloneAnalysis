import shutil
from datetime import datetime
import os

import requests
from bs4 import BeautifulSoup
from termcolor import colored


class Pixlr:
    def __init__(self):
        self.path = os.getcwd() + '/output/'
        self.headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
        }
        self.post_links = []
        self.next_page = None

    def log(self, text, color):
        current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(colored(f'| {current_time} |', 'white'), colored(f'{text}', color))

    def fetch_page(self, url):
        return requests.get(url, headers=self.headers).text

    def download_media(self, url, filename):
        with open(os.path.join(self.path, filename), 'wb') as f:
            res = requests.get(url, headers=self.headers, stream=True)
            shutil.copyfileobj(res.raw, f)

    def parse_content(self, content, album=False):
        soup = BeautifulSoup(content, 'html.parser')

        if album:
            self.log('Parsing current album page..', 'yellow')
            images = soup.find_all('div', class_='list-item-image fixed-size')
            self.post_links.extend(
                [img.find('a', class_='image-container --media')['href'] for img in images]
            )
            try:
                self.next_page = soup.find('a', attrs={'data-pagination': 'next'})['href']
            except TypeError:
                self.next_page = None
        else:
            self.log('Parsing current picture page..', 'yellow')
            return soup.find('input', id='embed-code-2', class_='text-input')['value']

    def process(self, url):
        self.log(f'Fetching: {url}', 'magenta')
        self.parse_content(self.fetch_page(url), album=True)

        while self.next_page:
            self.log(f'Fetching: {self.next_page}', 'magenta')
            self.parse_content(self.fetch_page(self.next_page), album=True)

        for post in self.post_links:
            picture_url = self.parse_content(self.fetch_page(post))
