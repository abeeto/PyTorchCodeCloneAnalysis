import requests
import time
from loguru import logger

from redisutility import RedisDriver

class PoeLeagueDriver:

    def __init__(self, config):

        #API setup
        self.conn = requests.Session()
        self.conn.headers = {
            'Authorization' : f"Bearer {config['apitokens']['league']}",
            'User-Agent' : config['settings']['apiuseragent']
        }

        #DB init
        self.db = RedisDriver(config)

    def filter_leagues(self, leagueId):
        '''Returns true if league is not one that we ignore'''

        if(leagueId != "Standard" and
        leagueId != "Hardcore" and
        leagueId != "SSF Standard" and
        leagueId != "SSF Hardcore"):
            return True
        
        return False

    def get_current_leagues(self):
        '''Returns a list of all active leagues in redis'''
        return self.db.get_current_leagues()

    def update_leagues(self):
        '''Retrieve the current active leagues from poe api'''

        #Make the api request
        req = self.conn.get("https://api.pathofexile.com/league")
        
        #Check for valid status code (Can implement more specialized stuff here later)
        if(req.status_code != 200):
            logger.info(f"Response received: {req.status_code}")

        #Pull the relevant data from the response
        reqbody = req.json()['leagues']

        #Iterate through the body and add the leagues
        currLeagues = []
        for league in reqbody:
            if(self.filter_leagues(league['id'])):
                self.db.add_league(league)
                currLeagues.append(league['id'])

        #After all leagues have been added, check and purge any that are not active
        self.db.sync_current_leagues(currLeagues)

        #Check and update indicies to ensure every active league has them set
        self.db.create_indicies()
