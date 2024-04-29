import os
import pygame as pg

vec = pg.math.Vector2

# Main window
TITLE = "dodgeNeat-PyTorch"
INFO_WIDTH = 0
INFO_HEIGHT = 600
PLAY_WIDTH = INFO_WIDTH + 800
PLAY_HEIGHT = INFO_HEIGHT
BLOCK_SIZE = 40
FPS = 60

# Directories
FILE_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(FILE_DIR, "../imgs")
SNAP_FOLDER = os.path.join(FILE_DIR, "../snapshots")

# Colors
GRID_COLOR = (40, 40, 40)
BACKGROUND_COLOR = (30, 30, 30)
SEP_LINE_COLOR = (60, 60, 60)
INFOS_COLOR = (255, 255, 255)

# Agent
X_AGENT = BLOCK_SIZE * 3.5
Y_AGENT = PLAY_HEIGHT // 2
AGENT_X_SPEED = BLOCK_SIZE // 2
AGENT_Y_SPEED = BLOCK_SIZE // 2

# Miscs
INFOS_SIZE = 20
Y_OFFSET_INFOS = 25
