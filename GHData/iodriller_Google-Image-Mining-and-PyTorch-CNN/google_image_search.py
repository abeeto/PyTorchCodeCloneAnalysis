################################################################################################################
# this is a modified version of the code from: "https://towardsdatascience.com/image-scraping-with-python-a96feda8af2d"
################################################################################################################

import os
import time
import requests
import io
from PIL import Image
import hashlib
from selenium import webdriver
import numpy as np


class google_image_search():
    def __init__(self,
                 min_count=10,
                 max_repeat_to_stop=5,
                 image_quality=85,
                 max_filename_len=10,
                 sleep_btw_interactions=1):
        self.min_count = min_count
        self.max_repeat_to_stop = max_repeat_to_stop
        self.image_quality = image_quality
        self.max_filename_len = max_filename_len
        self.sleep_btw_interactions = sleep_btw_interactions

    def scroll_to_end(self, wd, sleep_between_interactions):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

    def fetch_image_urls(self,
                         query: str,
                         max_links_to_fetch: int,
                         wd: webdriver,
                         sleep_between_interactions: int = 1):

        # build the google query
        search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

        # load the page
        wd.get(search_url.format(q=query))
        counter = []
        image_urls = set()
        image_count = 0
        results_start = 0
        while image_count < max_links_to_fetch:
            self.scroll_to_end(wd, sleep_between_interactions)

            # get all image thumbnail results
            thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
            number_results = len(thumbnail_results)

            print(
                f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}"
            )

            for img in thumbnail_results[results_start:number_results]:
                # try to click every thumbnail such that we can get the real image behind it
                try:
                    img.click()
                    time.sleep(sleep_between_interactions)
                except Exception:
                    continue

                # extract image urls
                actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
                for actual_image in actual_images:
                    if actual_image.get_attribute(
                            'src') and 'http' in actual_image.get_attribute('src'):
                        image_urls.add(actual_image.get_attribute('src'))

                image_count = len(image_urls)

                if len(image_urls) >= max_links_to_fetch:
                    print(f"Found: {len(image_urls)} image links, done!")
                    break
            else:
                print("Found:", len(image_urls),
                      "image links, looking for more ...")
                counter.append(image_count)
                load_more_button = wd.find_element_by_css_selector(".mye4qd")
                if load_more_button:
                    wd.execute_script(
                        "document.querySelector('.mye4qd').click();")
                    # if it does not find any new pictures, just return after 5 of the same
                    if len(counter) > self.min_count and np.sum(
                            np.diff(counter)[-self.max_repeat_to_stop:]):
                        print(
                            "does not find any new pictures and at the end, just return after 5 of the same"
                        )
                        return image_urls

            # move the result startpoint further down
            results_start = len(thumbnail_results)
        return image_urls

    def persist_image(self, folder_path: str, url: str):
        try:
            image_content = requests.get(url).content
        except Exception as e:
            print(f"ERROR - Could not download {url} - {e}")

        try:
            image_file = io.BytesIO(image_content)
            image = Image.open(image_file).convert('RGB')
            file_path = os.path.join(
                folder_path,
                hashlib.sha1(image_content).hexdigest()[:self.max_filename_len]
                + '.jpg')
            with open(file_path, 'wb') as f:
                image.save(f, "JPEG", quality=self.image_quality)
            print(f"SUCCESS - saved {url} - as {file_path}")
        except Exception as e:
            print(f"ERROR - Could not save {url} - {e}")

    def search_and_download(self,
                            search_term: str,
                            driver_path: str,
                            target_path='./images',
                            number_images=5):
        target_folder = os.path.join(target_path,
                                     '_'.join(search_term.lower().split(' ')))

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        with webdriver.Chrome(executable_path=driver_path) as wd:
            res = self.fetch_image_urls(
                search_term,
                number_images,
                wd=wd,
                sleep_between_interactions=self.sleep_btw_interactions)
        for elem in res:
            self.persist_image(target_folder, elem)
