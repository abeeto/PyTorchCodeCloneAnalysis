from bs4 import BeautifulSoup
from urllib import request

sauce = request.urlopen('https://pythonprogramming.net/parsememcparseface/').read()
soup = BeautifulSoup(sauce,'html5lib')
#print(soup.find_all('p'))

# for paragaraph in soup.find_all('p'):
#   print(paragaraph.text.strip())
#
# print(soup.get_text())
#
# print(soup.find_all('div',id='modalLogin'))
#
# nav = soup.nav
#
# for url in nav.find_all('a'):
#     print(url.get('href'))

# for div in soup.find_all('div', class_='body'):
#     print(div.text)

table = soup.table
table_rows = table.find_all('tr')

for tr in table_rows:
    td = tr.find_all('td')
    row = [i.text for i in td]
    print(row)