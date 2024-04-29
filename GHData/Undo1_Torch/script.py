from BeautifulSoup import BeautifulSoup
import mysql.connector
import config #Where we keep our passwords and stuff
import tldextract
import itertools

cnx = mysql.connector.connect(user=config.MySQLUsername(), password=config.MySQLPassword(), host=config.MySQLHost(), database=config.MySQLDatabase())
cursor = cnx.cursor()

query = ("SELECT Body, Score FROM Posts WHERE PostTypeId=2")

cursor.execute(query)

sites = []

for (Body, Score) in cursor:
	linksInAnswer = []
	soup = BeautifulSoup(Body)
	for link in soup.findAll('a'):
		extract = tldextract.extract(link.get('href'))
		# print extract
		if len(extract.subdomain) > 0:
			site = extract.subdomain + '.' + extract.domain + '.' + extract.suffix
		else:
			site = extract.domain + '.' + extract.suffix
		site = link.get('href')
		linksInAnswer.append(site)
	
	linksInAnswer = set(linksInAnswer)

	sites.extend(linksInAnswer)

groupedsites = [list(g) for k, g in itertools.groupby(sorted(sites))]

groupedsites = sorted(groupedsites, key=len, reverse=True)

for sitegroup in groupedsites:
	if len(sitegroup) > 3: print str(len(sitegroup)) + " x " + sitegroup[0]

cursor.close()
cnx.close()