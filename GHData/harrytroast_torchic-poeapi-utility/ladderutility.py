import requests
import threading
import time
from loguru import logger

from redisutility import RedisDriver

class PoeLadderDriver:

    def __init__(self, config):
        
        #API setup
        self.conn = requests.Session()
        self.conn.headers = {
            'Authorization' : f"Bearer {config['apitokens']['ladder']}",
            'User-Agent' : config['settings']['apiuseragent']
        }

        #DB init
        self.db = RedisDriver(config)

    def validate_consensus(self, data):
        '''Checks the timestamp on all api responses to ensure there were no
        updates to the api in the middle of the retrieval'''
        
        baseTS = data[0]['cached_since']
        for chunk in data:
            if(chunk['cached_since'] != baseTS):
                return False
        
        return True

    def get_ladder_chunk(self, leagueId, offset, data):
        
        req = self.conn.get(f"https://api.pathofexile.com/league/{leagueId}/ladder?limit=500&offset={500*offset}")
        if(req.status_code != 200):
            print(f"Uh uh got {req.status_code}")
            return
        reqbody = req.json()["ladder"]
        data[offset] = reqbody

    def get_ladder(self, leagueId):
        '''Given a leagueId, retrieve its current ladder from the poe api'''

        #Setup the data structs
        pthreadid = []
        data = [None] * 10 #shared data buffer for threads to load results into(safely)

        start = time.time()
        #Setup the threads
        for i in range(10):
            pthreadid.append(threading.Thread(target=self.get_ladder_chunk, args=(leagueId, i, data)))

        #Start the threads
        for i in range(10):
            pthreadid[i].start()

        #Wait for the threads
        for i in range(10):
            pthreadid[i].join()

        #Check to make sure data is all from the same time period
        if(not self.validate_consensus(data)):
            logger.error("Data mismatch")
            return -1

        logger.info(f"Took {time.time() - start} seconds to retrieve {leagueId}")

        start = time.time()
        #Load data into redis
        self.db.load_ladder_data(leagueId, data)
        logger.info(f"Took {time.time() - start} seconds to load {leagueId} data into redis")