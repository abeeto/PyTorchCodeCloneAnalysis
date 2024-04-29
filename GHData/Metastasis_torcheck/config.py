import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CONSENSUS_PATH = os.path.join(ROOT_DIR, os.path.normcase('data/cached-consensus'))

BLACKLIST_PATH = os.path.join(ROOT_DIR, os.path.normcase('data/blacklist'))

CLIENTLOG_PATH = os.path.join(ROOT_DIR, os.path.normcase('data/client-log'))

PEERS_PATH = os.path.join(ROOT_DIR, os.path.normcase('data/peers'))

TRACKED_CLIENTS_PATH = os.path.join(ROOT_DIR, os.path.normcase('data/tracked-clients'))
