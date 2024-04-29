import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent) + '/lib')
sys.path.append(str(Path(__file__).parent) + '/lib/face_sdk')
sys.path.append(str(Path(__file__).parent) + '/lib/face_sdk/models')
sys.path.append(str(Path(__file__).parent) + '/lib/face_sdk/models/face_recognition/face_recognition_arcface')
sys.path.append(str(Path(__file__).parent) + '/lib/stream_zmq')
sys.path.append(str(Path(__file__).parent) + '/web_app/video_server')
sys.path.append(str(Path(__file__).parent) + '/web_app')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_LOGGING_PATH = os.path.join(ROOT_DIR, 'config/logging.conf')

# print(ROOT_DIR, '\n')
# print(CONFIG_LOGGING_PATH, '\n')