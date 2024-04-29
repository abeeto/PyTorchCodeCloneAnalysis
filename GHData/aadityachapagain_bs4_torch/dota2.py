import requests
from bs4 import BeautifulSoup
import json

dota2_page = 'https://dota2.gamepedia.com/Heroes'

page = requests.get(dota2_page)

soup = BeautifulSoup(page.content,'html5lib')

heros_table = soup.find('div' ,id='mw-content-text').table

ability_list = [ability.text.strip() for ability in heros_table.find_all('th')]

heros_list = [[heros.text for heros in hero_span.find_all('span')] for hero_span in heros_table.find_all('td')]

heros_with_ability = list(zip(ability_list,heros_list))

with open('dota2.json', 'w') as fp:
    json.dump(heros_with_ability,fp)