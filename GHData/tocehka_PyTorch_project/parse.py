import requests
from bs4 import BeautifulSoup
from pathlib import Path
import urllib.request
import csv
import datetime

from utils import WriterManager
from config import *

conf = Config()
writer = WriterManager(file_name=conf.csv_file)
LABEL_CHECKER = "Артикул"
DELIMETR = "^"
# RESOURSE_CONN_VERIFY_STR = "Страница"
# RESOURSE_CONN_VERIFY_STR_PRIMAL = "Ваша надежная торговая марка"
PAGE_STEP = 24
GET_PARAMS = "?module=Catalog&ajax=1&cmd=showmore&from="
proxy = ProxyList()
proxies = proxy.get()
ALLOWED_STATUS = 200

#Resumption param
first_step = False

def get_page(url):
    try:
        r = requests.get(url, timeout=15)
        print(r.status_code)
        return r.text
    except:
        print("Resourse was out of timeout with default IP at " + datetime.datetime.now().isoformat())
        for proxy in proxies:
            proxy_item = {
                "http": "http://" + proxy,
                "https": "https://" + proxy
            }
            print(proxy)
            try:
                r = requests.get(url, proxies=proxy_item, timeout=5)
                print(r.status_code)
                if r.status_code != ALLOWED_STATUS:
                    raise Exception("Error HTTP code status was recieved: {}".format(r.status_code))
                return r.text
            except Exception as err:
                print(err)
                get_page(url)
            except requests.exceptions.Timeout:
                print("Resourse was out of timeout with " + proxy + " at " + datetime.datetime.now().isoformat())
                continue

def get_group_params(soup):
    try:
        quantity = soup.find("div", class_="block qty").find_all("b")
        return int(quantity[0].getText())
    except:
        print("0 items in group")
        return 0

def get_items_url(category):
    page = get_page(conf.base_url + category)
    soup = BeautifulSoup(page, conf.xml_preprocessor)
    quantity = get_group_params(soup)
    global first_step
    if first_step:
        first_step = False
        first_point = 250
    else:
        first_point = 0
    for step in range(first_point, 760, PAGE_STEP):
        page = get_page(conf.base_url + category + GET_PARAMS + str(step))
        soup = BeautifulSoup(page, conf.xml_preprocessor)
        items_field = soup.find("ul", class_="catalog_list_main").find_all("div", class_="title")
        for item in items_field:
            for url in item:
                get_item_info(url.get("href"), category)
    

def get_item_info(url, category):
    page = get_page(conf.base_url + url)
    soup = BeautifulSoup(page, conf.xml_preprocessor)
    parsed_item_attributes_title = soup.find("div", class_="item_params").find_all("div", class_="title")
    parsed_item_attributes = soup.find("div", class_="item_params").find_all("div", class_="body")
    name = soup.find("div", class_="top_nav").find_all("h1")
    item_attributes = []
    item_attributes.append("Наименование" + DELIMETR + name[0].getText())
    item_attributes.append("Категория" + DELIMETR + category.replace("/",""))
    label = ""
    if len(parsed_item_attributes_title) == len(parsed_item_attributes):
        for i in range(0, len(parsed_item_attributes)):
            title = parsed_item_attributes_title[i].getText().replace(":", "").strip()
            if title == LABEL_CHECKER:
                label = parsed_item_attributes[i].getText()
            item_attributes.append(title + DELIMETR + parsed_item_attributes[i].getText().strip())
    item_imgs = soup.find_all("a", class_="gal_zoom")
    item_attributes.append("images_path" + DELIMETR)
    for img in item_imgs:
        img_url = img.find("img").get("src").replace(conf.removing_part,'')
        if img == item_imgs[-1]:
            item_attributes[-1] = item_attributes[-1] + conf.img_dir + category + "/" + label + "/" + img_url.split("/")[-1]
            #get_item_image(label, category, img_url)
            break
        item_attributes[-1] = item_attributes[-1] + conf.img_dir + category + "/" + label + "/" + img_url.split("/")[-1] + ","
        #get_item_image(label, category, img_url)
    print(label)
    writer.write_row(item_attributes)

def get_item_image(label, category, img_url):
    label = "/" + label + "/"
    Path(conf.img_dir + category + "/" + label).mkdir(parents=True, exist_ok=True)
    print(conf.base_url + img_url)
    try:
        urllib.request.urlretrieve(conf.base_url + img_url, conf.img_dir + category + label + img_url.split("/")[-1])
    except:
        pass

def main():
    page = get_page(conf.base_url)
    soup = BeautifulSoup(page, conf.xml_preprocessor)
    all_links = soup.find("ul", class_="children").find_all("li")
    for link in all_links[:1]:
        if len(link.find("a").get("href").split("/")) <= 3:
            category = link.find("a").get("href")
            print(category + "--------------------------------")
            Path("./images/" + category.replace("/","")).mkdir(parents=True, exist_ok=True)
            get_items_url(category)

main()
del writer