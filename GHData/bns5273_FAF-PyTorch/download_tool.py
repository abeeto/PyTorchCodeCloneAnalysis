import json
import urllib.request


url = "https://api.faforever.com/data/gamePlayerStats?" \
      "fields[gamePlayerStats]=afterDeviation,afterMean,faction,score,startSpot,game" \
      "&filter=game.featuredMod.id==0;" \
              "game.mapVersion.id==560;" \
              "game.validity=='VALID';" \
              "scoreTime>'2000-01-01T12%3A00%3A00Z'" \
      "&page[size]=10000" \
      "&page[number]="
with open('setons.json', 'r') as infile:
    data = json.loads(infile.read())
    if download:
        for p in range(1, 10):  # pages. 87?
            with urllib.request.urlopen(url + str(p)) as j:
                print(url + str(p))
                new = json.loads(j.read())['data']
                data += new
            if new.__len__() == 0:
                break
        with open('setons.json', 'w') as outfile:
            json.dump(data, outfile)
