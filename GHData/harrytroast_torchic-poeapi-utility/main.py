import json
import time
from loguru import logger

from ladderutility import PoeLadderDriver
from leagueutility import PoeLeagueDriver

def loadConfig():
    '''Loads configuration file into memory'''
    fp = open('config.json')
    return json.load(fp)

def main():
    '''Handles the programs main execution loop'''

    #Load config
    config = loadConfig()
    
    #Create api objects
    leagues = PoeLeagueDriver(config)
    ladder = PoeLadderDriver(config)

    #Main loop
    while(1):

        #Update leagues
        leagues.update_leagues()

        #Retrieve the currently active leagues
        start = time.time()
        for league in leagues.get_current_leagues():
            ladder.get_ladder(league)
        
        logger.success(f"All leagues updated in {time.time() - start} seconds")
        logger.info(f"Waiting for 5 minutes...")
        time.sleep(300 - (time.time()-start))

if __name__ == "__main__":
    main()