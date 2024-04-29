import requests
import sys
import os
import datetime

class CowrieHFSScraper():

  def __init__(self):
    # This requires a local TOR service to be running
    self.proxies = {'http': "socks5://127.0.0.1:9150"}

  def scrapeURL(self, url):
   r = requests.get(url, proxies=self.proxies)

   # Check for HFS Server in HTTP response
   if "HFS" in r.headers["Server"]:

     # Pull helpful TAR archive of current site using built-in HFS function
     try:
       print("Downloading archive from " + url)
       archive = requests.get(url + "?mode=archive&recursive", proxies=self.proxies).content
     except:
       print("HFS Server Found. Download Failed")
       sys.exit()

     if archive:
       fname = url.split("http://")[1].strip("/").replace(".","_").replace(":","_") + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".tar"
       o = open(fname, "wb+")
       o.write(archive)
       o.close()
       print(fname + " Downloaded!")

C = CowrieHFSScraper()

if len(sys.argv) > 1:
  C.scrapeURL(sys.argv[1])

else:
  print("No URL specified for scraping")