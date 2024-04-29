import redis
import threading
from redis.commands.json.path import Path
from redis.commands.search.field import TagField, NumericField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from loguru import logger

class RedisDriver:

    def __init__(self, config):
        self.r = redis.Redis(
            host=config['redis']['host'],
            password=config['redis']['password'], 
            port=config['redis']['port'], 
            db=config['redis']['db'], 
            decode_responses=True)


    def create_indicies(self):
        schema = (
            NumericField("$.rank", as_name="rank"),
            TagField("$.dead", as_name="dead"),
            TagField("$.character.id", as_name="charid"),
            TagField("$.character.name", as_name="charname"),
            NumericField("$.character.level", as_name="charlvl"),
            TagField("$.character.class", as_name="charclass"),
            NumericField("$.character.time", as_name="chartime"),
            NumericField("$.character.score", as_name="charscore"),
            TagField("$.character.progress.UberEaterDefeated", as_name="charubereaterkilled"),
            TagField("$.character.progress.UberExarchDefeated", as_name="charuberexarchkilled"),
            TagField("$.character.progress.UberMavenDefeated", as_name="charubermavenkilled"),
            TagField("$.character.progress.UberSirusDefeated", as_name="charubersiruskilled"),
            TagField("$.character.progress.UberVenariusDefeated", as_name="charubervenariuskilled"),
            TagField("$.character.progress.UberUberElderDefeated", as_name="charuberuberelderkilled"),
            TagField("$.character.progress.UberShaperDefeated", as_name="charubershaperkilled"),
            NumericField("$.character.experience", as_name="charxp"),
            NumericField("$.character.depth.default", as_name="chardepthdefault"),
            NumericField("$.character.depth.solo", as_name="chardepthsolo"),
            TagField("$.account.name", as_name="accname"),
            TagField("$.account.realm", as_name="accrealm"),
            NumericField("$.account.challenges.total", as_name="acctotalchallenges"),
            TagField("$.account.twitch.name", as_name="acctwitch")
        )

        for league in self.get_current_leagues():
            try:
                op = self.r.ft(index_name=f'idx:{league}').create_index(schema, definition=IndexDefinition(prefix=[f"apijson:{league}:"], index_type=IndexType.JSON))
                logger.success(f"Index 'idx:{league}' created on 'apijson:{league}' and db returned {op}")
            except:
                logger.warning(f"Index 'idx:{league}' already exists for 'apijson:{league}'")
        

        
        

    ############################ LEAGUE OPERATIONS #############################################

    def add_league(self, leagueData):
        self.r.json().set(f'leagues:{leagueData["id"]}', '$', leagueData)
        self.r.sadd("leagues:active", f'{leagueData["id"]}')

    def get_current_leagues(self):
        activeleagues = self.r.smembers('leagues:active')
        return activeleagues
    
    def sync_current_leagues(self, currLeagues):
        '''Check for and remove leagues in the database that are no longer active'''

        databaseLeagues = self.r.smembers('leagues:active')

        #iterate through the database leagues and if they arent present in currLeagues, remove them
        for league in databaseLeagues:
            if(league not in currLeagues):
                self.r.srem('leagues:active', league)
                logger.info(f"League '{league}' was removed")
        
        logger.info("Leagues have been synced to the DB")

    #############################################################################################

    ############################ LADDER OPERATIONS ##############################################

    def load_ladder_chunk(self, leagueId, chunk, pipe):
        for entry in chunk['entries']:
            #Load the  json response entry for a character
            pipe.json().set(f'apijson:{leagueId}:{entry["rank"]}', '$', entry)

            #Push current experience to list
            pipe.lpush(f'charxp:{leagueId}:{entry["character"]["id"]}', entry["character"]["experience"])
            pipe.ltrim(f'charxp:{leagueId}:{entry["character"]["id"]}', 0, 9)

            #Push current rank to list
            pipe.lpush(f'charrank:{leagueId}:{entry["character"]["id"]}', entry["rank"])
            pipe.ltrim(f'charrank:{leagueId}:{entry["character"]["id"]}', 0, 9)

    def load_ladder_data(self, leagueId, data):
        '''Given leagueId and the api data from it, load it all into redis'''
        
        pipe = self.r.pipeline()

        pthreadid = []

        #Setup threads
        for i in range(10):
            pthreadid.append(threading.Thread(target=self.load_ladder_chunk, args=(leagueId, data[i], pipe)))

        #Start threads
        for i in range(10):
            pthreadid[i].start()

        #Wait for threads
        for i in range(10):
            pthreadid[i].join()

        #Execute pipeline transaction
        pipe.execute()