#!/usr/bin/env python

import getopt
import sys
import os
import traceback
import random
import logging
import yaml
import signal

from threading import Thread

from torch_song.torch_song import TorchSong
from torch_song.songbook.songbook_manager import SongbookManager
from torch_song.server.control_udp_server import TorchControlServer
from torch_song.icosahedron.icosahedron import Icosahedron
from torch_song.edge.edge_handlers import *

songbooks = [
    'songbooks/eli_eli.yml'
    'songbooks/rainy.yml'
    'songbooks/out_of_sync.yml'
]

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hscv", ["help", "sim", "calibrate", "verbose"])
    except getopt.GetoptError:
        print('Unrecognized option')
        sys.exit(2)

    # Opts
    sim = False
    calibrate = False
    verbose = False
    for opt, arg in opts:
        if opt in ('-s', '--sim'):
            sim = True
        if opt in ('-c', '--cal'):
            calibrate = True
        if opt in ('-v', '--verbose'):
            verbose = True

    # Build config
    try:
        stream = open('conf/default-mod.yml', 'r')
    except Exception:
        stream = open('conf/default.yml', 'r')
    config = yaml.load(stream)

    # Setup loggers
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    loggingPort = config['logging']['remote_port']

    socketEdgeHandler = SocketEdgeHandler('localhost', loggingPort)
    socketEdgeHandler.createSocket()
    socketEdgeHandler.setLevel(logging.INFO)

    streamHandler = EdgeStreamHandler(sys.stdout)
    streamHandler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    streamHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)
    logger.addHandler(socketEdgeHandler)

    ts = None
    sbm = None
    cs_server = None
    error_code = 0

    try:
        # Create torch song
        ts = TorchSong(config=config, num_edges=config['num_edges'], sim=sim, verbose=verbose)
        sbm = SongbookManager(songbooks, ts, config['songbook_mode'])

        # Start TorchSong server
        cs_local_port = config['control_server']['local_port']
        cs_remote_port = config['control_server']['remote_port']
        cs_server = TorchControlServer(cs_local_port, cs_remote_port, ts, sbm)

        cs_server_thread = Thread(target=cs_server.serve_forever)
        cs_server_thread.daemon = True
        cs_server_thread.start()

        signal.signal(signal.SIGTERM, lambda sig, frame: sbm.kill())

        if (calibrate):
            ts.calibrate()
        sbm.run()
    except KeyboardInterrupt:
        logging.info('Received ctrl-c, cleaning up')
    except Exception as e:
        logging.error(traceback.format_exc())
        error_code = 1
    finally:
        logging.info('Closing shop')
        try:
            if sbm is not None:
                sbm.kill()
            if ts is not None:
                ts.kill()
            if cs_server is not None:
                cs_server.kill()
            socketEdgeHandler.close()
        except Exception as e:
            print(e)
        sys.exit(error_code)


if __name__ == '__main__':
    main()
