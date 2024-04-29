import requests
from bs4 import BeautifulSoup

#decleare the url
quote_page = 'https://www.bloomberg.com/quote/SPX:IND'

#qury the webpage and return the webpage to the variable name page
page = requests.get(quote_page)

#parse the html using the beautiful soup and store in variable named soup
soup = BeautifulSoup(page.content,'html5lib')
print(soup)
name_box = soup.find_all('p')
for names in name_box:
    print(names.text)