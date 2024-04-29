import yaml
from pathlib import Path
from icrawler.builtin import GoogleImageCrawler
from icrawler.downloader import ImageDownloader
import pandas as pd
from threading import RLock
import os


datasources = []

class CustomizedDownloader(ImageDownloader):

  def __init__(self, thread_num, signal, session, storage, process_meta_func):
    super().__init__(thread_num, signal, session, storage)
    self.process_meta_func = process_meta_func

  def process_meta(self, task):
    self.process_meta_func(task)


class Datasource():

  def __init__(self, platform, keywords, storage):
    self.platform = platform
    self.storage = Path(storage)
    self.keywords = keywords

  def crawl(self):
    for entity in self.keywords:
      if isinstance(entity, dict):
        keyword = entity.get('search')
        annotation = entity.get('annotation', entity)
        maxnum = entity.get('maxnum', 100)
      else:
        keyword = annotation = entity
        maxnum = 100
      self.crawl_keyword(keyword, annotation, maxnum)

  def crawl_keyword(self, keyword, annotation, maxnum):
      crawler = self.create_crawler(keyword, annotation)
      crawler.crawl(keyword, max_num=maxnum)

  def create_crawler(self, keyword, annotation):
    if self.platform == 'google':
      return GoogleImageCrawler(
        storage={ 'root_dir': self.storage / self.platform / keyword },
        downloader_cls=CustomizedDownloader,
        extra_downloader_args={
          'process_meta_func': self.create_annotation_hook(keyword, annotation)
        }
      )

  def create_annotation_hook(self, keyword, annotation):
    def annotation_hook(task):
      nonlocal self, keyword
      global lock, annotation_file

      if not task.get('success'):
        return
      with lock:
        df = pd.DataFrame.from_records([{
          'url': task.get('file_url'),
          'image': self.storage / self.platform / keyword / task.get('filename'),
          'class': annotation
        }])
        df.to_csv(annotation_file, mode='a', header=False)

    return annotation_hook


if __name__ == '__main__':
  with open('config.yml', 'r') as f:
    config = yaml.load(f)
    annotation_file = Path(config['storage']) / 'annotation.csv'
    lock = RLock()

    if os.path.isfile(annotation_file):
      annotation_df = pd.read_csv(annotation_file)
    else:
      annotation_df = pd.DataFrame(columns=['url', 'image', 'class'])
      annotation_df.to_csv(annotation_file)

    for datasource in config['datasource']:
      datasources.append(
        Datasource(
          datasource['engine'],
          datasource['keywords'],
          config['storage']
        )
      )
    datasources[0].crawl()

