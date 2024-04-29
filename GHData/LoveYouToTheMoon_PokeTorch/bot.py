import time
from threading import Thread
import os
import discord
import random
import json
import sys
from yolov5.detect import run
from twocaptcha import TwoCaptcha
import config as cfg
import asyncio
import aiohttp        
import aiofiles
import requests

inventory = dict()
client = discord.Client()


def retrieve():
	headers = {
		"authorization": cfg.auth
	}
	r = requests.get(f"https://discord.com/api/v9/channels/{cfg.channel}/messages?limit=1", headers=headers)
	jsonn = json.loads(r.text)
	return jsonn[0]

#determine pokeball by rarity
def throwpoke(rarity):
  switcher = {
    "Common": cfg.commonthrow,
    "Uncommon": cfg.uncommonthrow,
    "Rare": cfg.rarethrow,
    "Super Rare": cfg.superrarethrow,
    "Legendary": cfg.legendarythrow,
    "Shiny": cfg.shinythrow,
    "Golden": cfg.shinythrow,
  }
  if switcher.get(rarity, "pb") == "pb":
    inventory["pokeball"] -= 1
  elif switcher.get(rarity, "pb") == "gb":
    inventory["greatball"] -= 1
  elif switcher.get(rarity, "pb") == "ub":
    inventory["ultraball"] -= 1
  elif switcher.get(rarity, "pb") == "mb":
    inventory["masterball"] -= 1

  return switcher.get(rarity, "pb")


def sleep(sec):
  if cfg.farm_captcha:
    return sec
    #print("slept",sec,"seconds")
  else:
    seconds = random.randint(sec*1000,(sec+2)*1000)
    milliseconds = 0.001*seconds
    return milliseconds

async def ptsolve(url):
  last = url.split('/')[-1]
  print(f"Downloading image {last}.")
  async with aiohttp.ClientSession() as session:
    async with session.get(url) as resp:
      if resp.status == 200:
        f = await aiofiles.open(f'captchaimg/{last}', mode='wb')
        await f.write(await resp.read())
        await f.close()
  print("Finished downloading.")
  solve = run(source=f'captchaimg/{last}')
  solve.sort(key=lambda x: x[1])
  res = ""
  for i in solve:
    res+=i[0]
  print(f"Model returned {res}.")
  return res


def updateinv(arr):
  inventory.update({"pokecoin":int(arr[0])})
  inventory.update({"fishingtoken":int(arr[1])})
  inventory.update({"votecoin":int(arr[2])})
  inventory.update({"tickets":int(arr[3])})
  inventory.update({"pokeball":int(arr[6])})
  inventory.update({"greatball":int(arr[7])})
  inventory.update({"ultraball":int(arr[8])})
  inventory.update({"diveball":int(arr[9])})
  inventory.update({"masterball":int(arr[10])})
  '''
  inventory.update({"rare_lootbox":int(arr[13])})
  inventory.update({"superrare_lootbox":int(arr[14])})
  inventory.update({"legendary_lootbox":int(arr[15])})
  inventory.update({"shiny_lootbox":int(arr[16])})
  inventory.update({"premierball":int(arr[18])})
  inventory.update({"eonticket":int(arr[19])})
  inventory.update({"douse_drive":int(arr[20])})
  inventory.update({"repel":int(arr[23])})
  inventory.update({"super_repel":int(arr[24])})
  inventory.update({"max_repel":int(arr[25])})
  inventory.update({"honey":int(arr[26])})
  inventory.update({"goldenrazzberry":int(arr[27])})
  inventory.update({"lootbox":int(arr[28])})
  inventory.update({"poke_egg":int(arr[29])})
  inventory.update({"pokelure":int(arr[32])})
  inventory.update({"mistys_lure":int(arr[33])})
  inventory.update({"sea_flute":int(arr[34])})
  inventory.update({"candy":int(arr[36])})
  inventory.update({"lollipop":int(arr[37])})
  inventory.update({"chocolate_bar":int(arr[38])})
  inventory.update({"broccoli":int(arr[39])})
  inventory.update({"candy_cane":int(arr[40])})
  inventory.update({"cupcake":int(arr[41])})
  inventory.update({"gingerbread_cookie":int(arr[42])})
  '''
  
@client.event
async def on_ready():
  # login 
  print("We have logged on as {0.user}".format(client))

  # get channel
  channel = client.get_channel(int(cfg.channel))
  m = await channel.history(limit=1).flatten()

  # check for captcha
  if m[0].attachments != []:
    url = m[0].attachments[0].url
    await channel.send(await ptsolve(url))

    # if model fails use 2captcha, else exit
    for _ in range(10):
      m = await channel.history(limit=1).flatten()
      if "thank you, you may continue hunting" in m[0].content:
        break
      if cfg.captcha != None:
        solver = TwoCaptcha(cfg.captcha)
        result = solver.normal(url,numeric=1)
        await channel.send(str(result["code"]))
        await asyncio.sleep(sleep(5))
      else:
        print("Failed to solve captcha.")
        exit(0)

  # get inventory and update dictionary

  print("Checking inventory.")
  await channel.send(";inv")
  await asyncio.sleep(3)
  m = await channel.history(limit=1).flatten()  
  emb = m[0].embeds[0].fields[0].value.split("\n")
  emb.extend(m[0].embeds[0].fields[1].value.split("\n"))
  emb.extend(m[0].embeds[0].fields[2].value.split("\n"))
  for i in range(len(emb)):
    res=emb[i][emb[i].find("**")+len("**"):emb[i].rfind("**")]
    emb[i] = res.replace(",","")
  for i in emb:
    if "uses left" in i:
      emb.remove(i)
  updateinv(emb)
  print("Inventory recieved.")
  await asyncio.sleep(sleep(2))

  # main loop
  for _ in range(10000): 
    # encounter pokemon
    print("Encountering Pokemon.")
    await channel.send(";p")
    await asyncio.sleep(sleep(3))

    # solve captcha
    m = await channel.history(limit=1).flatten()
    if m[0].attachments != []:
      url = m[0].attachments[0].url
      await channel.send(await ptsolve(url))
      await asyncio.sleep(sleep(3))
      for _ in range(10):
        m = await channel.history(limit=1).flatten()
        if "thank you, you may continue hunting" in m[0].content:
          break
        if cfg.captcha != None:
          solver = TwoCaptcha(cfg.captcha)
          result = solver.normal(url,numeric=1)
          await channel.send(str(result["code"]))
          await asyncio.sleep(sleep(5))
        else:
          print("Failed to solve captcha.")
          exit(0)
      continue

    # identify pokemon rarity
    pokemon = m[0].content.split("**")[3]
    rarity = m[0].embeds[0].footer.text.split()[0]
    if rarity == "Super":
      rarity = "Super Rare"
    print(f"You found a {rarity} {pokemon}.")

    # throw pokeball
    await channel.send(throwpoke(rarity))
    await asyncio.sleep(sleep(2))

    # check if pokemon was caught
    m = await channel.history(limit=2).flatten()

    if "You caught a" in m[1].embeds[0].description:
      print(f"Successfully caught a {pokemon}!")
    else: 
      print(f"{pokemon} broke out.")

    # if pokeballs are out, buy
    if inventory["pokeball"] == 0:
      if inventory["pokecoin"] >= cfg.buypb*200:
        await channel.send(f";shop buy 1 {str(cfg.buypb)}")
        inventory["pokecoin"] -= cfg.buypb*200
        inventory["pokeball"] = cfg.buypb
        print(f"Bought {str(cfg.buypb)} pokeballs for {str(cfg.buypb*200)} pokecoins.")
    elif inventory["greatball"] == 0:
      if inventory["pokecoin"] >= cfg.buygb*500:
        await channel.send(f";shop buy 2 {str(cfg.buygb)}")
        inventory["pokecoin"] -= cfg.buygb*500
        inventory["greatball"] = cfg.buygb
        print(f"Bought {str(cfg.buygb)} greatballs for {str(cfg.buygb*500)} pokecoins.")
    elif inventory["ultraball"] == 0:
      if inventory["pokecoin"] >= cfg.buyub*1500:
        await channel.send(f";shop buy 3 {str(cfg.buyub)}")
        inventory["pokecoin"] -= cfg.buyub*1500
        inventory["ultraball"] = cfg.buyub
        print(f"Bought {str(cfg.buyub)} ultraballs for {str(cfg.buyub*1500)} pokecoins.")
    elif inventory["masterball"] == 0:
      if inventory["pokecoin"] >= cfg.buymb*100000:
        await channel.send(f";shop buy 4 {str(cfg.buymb)}")
        inventory["pokecoin"] -= cfg.buymb*100000
        inventory["masterball"] = cfg.buymb
        print(f"Bought {str(cfg.buymb)} masterballs for {str(cfg.buymb*100000)} pokecoins.")
    await asyncio.sleep(sleep(8))

def start():
  
  client.run(cfg.auth, bot=False)
