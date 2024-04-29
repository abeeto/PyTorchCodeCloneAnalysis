"""Implements the game loop and handles the user's events."""

import os
import random
import numpy as np
import pygame as pg
import constants as const

from utils import message, distance

vec = pg.math.Vector2
n_snap = 0

# Manually places the window
os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (50, 50)

STATE_SPACE = 13
ACTION_SPACE = 4

MAX_FRAME = 250

REWARD_CLOSE_FOOD = 1
REWARD_EAT = 10

PENALTY_WANDER = -1
PENALTY_COLLISION = -10
PENALTY_FAR_FOOD = 0

MOVES = {0: "right", 1: "left", 2: "down", 3: "up"}


class Block(pg.sprite.Sprite):
    def __init__(self, x, y, w, h, color):
        pg.sprite.Sprite.__init__(self)

        self.pos = vec(x, y)
        self.color = color
        self.image = pg.Surface((w, h))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.center = self.pos

    def draw(self, screen):
        pg.draw.rect(screen, self.color, self.rect)


class Game:
    def __init__(self, human=False, grid=False, infos=True) -> None:
        pg.init()
        self.human = human
        self.grid = grid
        self.infos = infos
        self.screen = pg.display.set_mode([const.PLAY_WIDTH, const.PLAY_HEIGHT])
        self.clock = pg.time.Clock()

        pg.display.set_caption(const.TITLE)

        self.running = True
        self.state_space = STATE_SPACE
        self.action_space = ACTION_SPACE
        self.n_games = 0
        self.n_frames_threshold = 0
        self.score = 0
        self.highest_score = 0
        self.sum_scores = 0
        self.sum_rewards = 0
        self.mean_scores = [0]
        self.mean_rewards = [0]
        self.reward_episode = 0

        self.player = Block(0, 0, const.BLOCK_SIZE, const.BLOCK_SIZE, pg.Color("Blue"))
        self.food = Block(0, 0, const.BLOCK_SIZE, const.BLOCK_SIZE, pg.Color("Green"))
        self.enemies = [
            Block(
                (const.INFO_WIDTH + const.PLAY_WIDTH) // 2 - const.BLOCK_SIZE // 2,
                const.PLAY_HEIGHT // 2,
                const.BLOCK_SIZE * 1,
                const.BLOCK_SIZE * 9,
                pg.Color("Red"),
            ),
            # Block(
            #     (const.INFO_WIDTH + const.PLAY_WIDTH) // 2 - const.BLOCK_SIZE // 2,
            #     const.PLAY_HEIGHT // 2,
            #     const.BLOCK_SIZE * 9,
            #     const.BLOCK_SIZE * 1,
            #     pg.Color("Red"),
            # ),
        ]

        self.place_player()
        self.place_food()

        self.direction = None
        self.dangerous_locations = set()
        self.distance_food = distance(self.player.pos, self.food.pos)

    ####### Methods #######

    def random_coordinates(self):
        idx_x = random.randint(
            1, ((const.PLAY_WIDTH - const.INFO_WIDTH) // const.BLOCK_SIZE) - 1
        )
        idx_y = random.randint(1, (const.PLAY_HEIGHT // const.BLOCK_SIZE) - 1)
        x = const.INFO_WIDTH + idx_x * const.BLOCK_SIZE
        y = idx_y * const.BLOCK_SIZE
        
        return x, y

    def place_player(self):
        x, y = self.random_coordinates()

        self.player.pos = vec(x, y)
        self.player.rect.center = self.player.pos
        
        obstacles = [enemy.rect for enemy in self.enemies] + [self.food.rect]
        if self.player.rect.collidelist(obstacles) != -1:
            self.place_player()

    def place_food(self):
        x, y = self.random_coordinates()

        self.food.pos = vec(x, y)
        self.food.rect.center = self.food.pos

        # Checking for potential collisions with other entities
        obstacles = [enemy.rect for enemy in self.enemies] + [self.player.rect]
        if self.food.rect.collidelist(obstacles) != -1:
            self.place_food()

    def reset(self) -> np.array:
        """Resets the game and return its corresponding state."""
        self.score = 0
        self.n_frames_threshold = 0
        self.reward_episode = 0
        self.dangerous_locations.clear()
        self.place_player()
        self.place_food()

        return self.get_state()

    def move(self, action) -> None:
        """
        Moves player according to the action chosen by the model.

        args:
            action (int, required): action chosen by the human/agent to move the player
        """

        if self.human:
            keys = pg.key.get_pressed()  # Keyboard events
            if keys[pg.K_RIGHT]:
                self.direction = "right"
                self.player.pos.x += const.AGENT_X_SPEED
            elif keys[pg.K_LEFT]:
                self.direction = "left"
                self.player.pos.x += -const.AGENT_X_SPEED
            elif keys[pg.K_UP]:
                self.direction = "up"
                self.player.pos.y += -const.AGENT_Y_SPEED
            elif keys[pg.K_DOWN]:
                self.direction = "down"
                self.player.pos.y += const.AGENT_Y_SPEED
        else:
            self.direction = MOVES[action]
            if self.direction == "right":  # going right
                self.player.pos.x += const.AGENT_X_SPEED
            elif self.direction == "left":  # going left
                self.player.pos.x += -const.AGENT_X_SPEED
            elif self.direction == "up":  # going down
                self.player.pos.y += -const.AGENT_Y_SPEED
            elif self.direction == "down":  # going up
                self.player.pos.y += const.AGENT_Y_SPEED

        # Updating pos
        self.player.rect.center = self.player.pos

    def step(self, action):
        self.n_frames_threshold += 1

        self.events()
        self.move(action)

        reward, done = self.get_reward()
        state = self.get_state()

        return state, reward, done

    def get_state(self) -> np.array:

        r_player, r_food = self.player.rect, self.food.rect
        state = [
                # current direction
                self.direction == "right",
                self.direction == "left",
                self.direction == "down",
                self.direction == "up",
                # food relative position
                r_player.right <= r_food.left,  # food is right
                r_player.left >= r_food.right,  # food is left
                r_player.bottom <= r_food.top,  # food is bottom
                r_player.top >= r_food.bottom,  # food is up
                # dangers
                self.wall_collision(offset=const.BLOCK_SIZE),
            ]
        enemy_dangers = self.enemy_danger()
        state.extend(enemy_dangers)
        
        return np.array(state, dtype=np.float32)

    def get_reward(self) -> tuple:
        done = False
        reward = 0

        # stops episode if the player does nothing but wonder around
        if self.n_frames_threshold > MAX_FRAME:
            return PENALTY_WANDER, True

        # checking for failure (wall or enemy collision)
        if self.wall_collision(offset=0) or self.enemy_collision():
            return PENALTY_COLLISION, True

        # checking if player is getting closer to food
        self.old_distance_food = self.distance_food
        self.distance_food = distance(self.player.pos, self.food.pos)
        if self.distance_food < self.old_distance_food:
            reward = REWARD_CLOSE_FOOD

        # checking for any enemy nearby and tag its location as dangerous
        if any(self.enemy_danger()):
            if self.player.rect.center not in self.dangerous_locations:
                self.dangerous_locations.add(self.player.rect.center)
                reward = 1
            else:
                reward = -1

        # checking if eat:
        if self.food_collision():
            self.score += 1
            self.n_frames_threshold = 0
            self.place_food()
            reward = REWARD_EAT

        return reward, done

    def wall_collision(self, offset):
        r = self.player.rect
        return (
            r.left - offset < const.INFO_WIDTH
            or r.right + offset > const.PLAY_WIDTH
            or r.top - offset < 0
            or r.bottom + offset > const.PLAY_HEIGHT
        )

    def enemy_collision(self):
        return bool(pg.sprite.spritecollide(self.player, self.enemies, False))
    
    def enemy_danger(self):
        # checking left, right, up and down
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dangers = [0] * self.action_space

        for enemy in self.enemies:
            for count, offset in enumerate(offsets):
                buffer_r = self.player.rect.copy().move(offset)
                if buffer_r.colliderect(enemy.rect):
                    dangers[count] = 1

        return dangers

    def food_collision(self):
        return self.player.rect.colliderect(self.food.rect)

    def events(self):
        for event in pg.event.get():
            if (
                event.type == pg.QUIT
                or event.type == pg.KEYDOWN
                and event.key == pg.K_q
            ):
                self.running = False

    def render(self):
        """TODO"""

        self.screen.fill(const.BACKGROUND_COLOR)
        self.draw_entities()

        if self.grid:
            self.draw_grid()

        if self.infos:
            self.draw_infos()

        pg.display.flip()
        self.clock.tick(const.FPS)

    def draw_entities(self):
        """TODO"""
        self.player.draw(self.screen)
        self.food.draw(self.screen)

        for enemy in self.enemies:
            enemy.draw(self.screen)

    def draw_grid(self):
        """TODO"""
        for i in range(1, const.PLAY_WIDTH // const.BLOCK_SIZE):
            # vertical lines
            p_v1 = const.INFO_WIDTH + const.BLOCK_SIZE * i, 0
            p_v2 = const.INFO_WIDTH + const.BLOCK_SIZE * i, const.PLAY_HEIGHT

            # horizontal lines
            p_h1 = 0, const.BLOCK_SIZE * i
            p_h2 = const.PLAY_WIDTH, const.BLOCK_SIZE * i

            pg.draw.line(self.screen, const.GRID_COLOR, p_v1, p_v2)
            pg.draw.line(self.screen, const.GRID_COLOR, p_h1, p_h2)

    def draw_infos(self):
        """Draws game informations"""

        if self.score > self.highest_score:
            self.highest_score = self.score

        perc_exploration = (
            self.agent.n_exploration
            / (self.agent.n_exploration + self.agent.n_exploitation)
            * 100
        )
        perc_exploitation = (
            self.agent.n_exploitation
            / (self.agent.n_exploration + self.agent.n_exploitation)
            * 100
        )
        perc_threshold = int((self.n_frames_threshold / MAX_FRAME) * 100)

        infos = [
            f"Game: {self.n_games}",
            f"Reward game: {round(self.reward_episode, 1)}",
            f"Mean reward: {round(self.mean_rewards[-1], 1)}",
            f"Score: {self.score}",
            f"Highest score: {self.highest_score}",
            f"Mean score: {round(self.mean_scores[-1], 1)}",
            f"Initial Epsilon: {self.agent.max_epsilon}",
            f"Epsilon: {round(self.agent.epsilon, 4)}",
            f"Exploration: {round(perc_exploration, 3)}%",
            f"Exploitation: {round(perc_exploitation, 3)}%",
            f"Last decision: {self.agent.last_decision}",
            f"Threshold: {perc_threshold}%",
            f"Time: {int(pg.time.get_ticks() / 1e3)}s",
            f"FPS: {int(self.clock.get_fps())}",
        ]

        # Drawing infos
        for i, info in enumerate(infos):
            message(
                self.screen,
                info,
                const.INFOS_SIZE,
                const.INFOS_COLOR,
                (5, 5 + i * const.Y_OFFSET_INFOS),
            )

        # sep line
        pg.draw.line(
            self.screen,
            const.SEP_LINE_COLOR,
            (const.INFO_WIDTH, 0),
            (const.INFO_WIDTH, const.INFO_HEIGHT),
        )


def set_global_seed(seed: int) -> None:
    """
    Sets random seed into PyTorch, numpy and random.

    Args:
        seed: random seed
    """

    try:
        import torch
    except ImportError:
        print("Module PyTorch cannot be imported")
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)


def main():
    pass


if __name__ == "__main__":
    main()
