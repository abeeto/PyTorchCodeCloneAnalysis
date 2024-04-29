import os
import re
import requests
from bs4 import BeautifulSoup


site = input("Enter the website:")
dir = input("Enter the directory:")

# site = 'https://images.search.yahoo.com/search/images;_ylt=AwrEzOAjRixe1GAAWXeJzbkF;_ylu=X3oDMTBsZ29xY3ZzBHNlYwNzZWFyY2gEc2xrA2J1dHRvbg--;_ylc=X1MDOTYwNjI4NTcEX3IDMgRhY3RuA2NsawRjc3JjcHZpZANWYkRIdmpFd0xqS1ZJU3NUWGl4R0dBTEVOelF1TndBQUFBQVJORnZKBGZyA3lmcC10BGZyMgNzYS1ncARncHJpZANLRmVrSFBDblNnT25wcjRJZmIzOUxBBG5fc3VnZwMxMARvcmlnaW4DaW1hZ2VzLnNlYXJjaC55YWhvby5jb20EcG9zAzAEcHFzdHIDBHBxc3RybAMEcXN0cmwDNwRxdWVyeQNwdXBwaWVzBHRfc3RtcAMxNTc5OTU5NzE1?p=puppies&fr=yfp-t&fr2=sb-top-images.search&ei=UTF-8&n=60&x=wrt'
# dir = '/home/andrew/PycharmProjects/PyTorch/test'
response = requests.get(site)

soup = BeautifulSoup(response.text, 'html.parser')
img_tags = soup.find_all('img')

urls=[]

for img in img_tags:
    try:
        imgsrc = [img['src']]
    except:
        continue
    else:
        [imgsrc] = imgsrc
        urls.append(imgsrc)


for url in urls:
    filename = re.search(r'/([\w_-]+[.](jpg|gif|png))$', url)
    f = open(filename.group(1), 'wb')
    if 'http' not in url:
        url = '{}{}'.format(site, url)
    response = requests.get(url)
    f.write(os.path.join(dir, response.content))