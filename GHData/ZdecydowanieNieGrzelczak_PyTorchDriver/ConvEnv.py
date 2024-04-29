import math
import random, time
import numpy as np
from math import trunc
import copy

class Scribe:

    def __init__(self, nr_of_quests, nr_of_stations, nr_of_locations, is_total=False):
        self.picked = np.zeros(shape=nr_of_quests)
        self.ended = np.zeros(shape=nr_of_quests)
        self.tanked = 0
        self.mobility = np.zeros(shape=nr_of_locations)
        self.died = np.zeros(shape=nr_of_locations)
        self.is_total = is_total
        self.invalid_action = 0
        self.steps = 0
        self.percentage = 0

    def __str__(self):
        if self.is_total:
            representation = "During whole game:\n"
        else:
            representation = "During run:\n"
        # picked = "Picked: {0:.0f} Q1: {p[0]:.0f} Q2: {p[1]:.0f} Q3: {p[2]:.0f} Q4: {p[3]:.0f} Q5: {p[4]:.0f}\n".format(np.sum(self.picked), p=self.picked)
        picked = "Picked: {0:.0f} \n".format(np.sum(self.picked))
        # ended = "Ended:  {0:.0f} Q1: {p[0]:.0f} Q2: {p[1]:.0f} Q3: {p[2]:.0f} Q4: {p[3]:.0f} Q5: {p[4]:.0f}\n".format(np.sum(self.ended), p=self.ended)
        ended = "Ended:  {0:.0f}\n".format(np.sum(self.ended))
        tanked = "Tanked: {0:.0f} \n".format(self.tanked)
        if self.steps == 0:
            invalid = "Invalid actions taken: {0}\n".format(self.invalid_action)
        else:
            invalid = "Invalid actions taken: {0} | {1}\nPercentage:{2:.2f}%".format(self.invalid_action, self.steps,
                                                                                     self.percentage)

        return ''.join([representation, picked, ended, tanked, invalid])

    def __add__(self, other):
        self.picked += other.picked
        self.ended += other.ended
        self.tanked += other.tanked
        self.died += other.died
        self.invalid_action += other.invalid_action

    def set_steps(self, steps):
        self.steps = steps
        self.percentage = self.invalid_action / self.steps * 100




class ConvEnv:

    wait_discount = 1
    gas_max = 200
    map_size = 15
    prepaid = 0.1
    gas_price = 0

    reward_per_step = 15
    large_quest_bonus = 1.07
    random_deviation = 0.2
    quest_multiplier = 1
    reward = 0
    start_gas = 200
    death_reward = 1
    start_money = 500
    is_done = False
    action_space = (0, 1, 2, 3)
    reward_normalizer = 10
    quest_reward = 10

    station_code = 50
    quest_code = 100
    reward_code = 180
    player_code = 255


    def __init__(self, quest_nr=5, station_nr=3, width=15, height=15, uniform_gas_stations=False, normalize_rewards=False):
        self.width = width
        self.height = height
        self.quest_nr = quest_nr
        self.station_nr = station_nr
        self.gas = self.start_gas
        self.normalize_reward = normalize_rewards
        self.money = self.start_money
        self.uniform_gas_stations = uniform_gas_stations
        self.observation_space = self.width * self.height + 2
        self.cargo = [0 for i in range(quest_nr)]
        self.rewards = [0 for i in range(quest_nr)]

        self.uniform_gas_list = [ [int(width / 2), int(height / 2)], [int(width / 4), int(height / 2)],
                                  [int(width / 2) + 2, int(height / 4 * 3)], [int(width / 8), int(height / 3 + 1)],
                                  [int(width / 5 * 4), int(height / 5 * 3)], [int(width / 4 * 3), int(height / 5 * 3)]]

        if self.height == 15 and self.width == 15:
            self.uniform_gas_list = [[2, 5], [10, 3], [5, 11], [13, 12], [7, 8], [1, 13], [0, 0]]

        random.seed(time.time())
        self.map = np.zeros(shape=(self.width, self.height)) #, dtype=np.uint8)
        self.player_pos = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
        self.setup_dict = self.set_random_points(self.player_pos)

        self.scribe = Scribe(self.quest_nr, self.station_nr, self.map_size ** 2)

        self.actions = [self.action_up, self.action_down, self.action_left, self.action_right, self.action_special]
        self.action_count = len(self.action_space)


    def reset(self):
        self.map = np.zeros(shape=(self.width, self.height)) #, dtype=np.uint8)
        self.player_pos = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
        self.setup_dict = self.set_random_points(self.player_pos)
        self.gas = self.start_gas
        self.money = self.start_money
        self.cargo = [0 for i in range(self.quest_nr)]
        self.rewards = [0 for i in range(self.quest_nr)]
        self.is_done = False
        self.reward = 0
        self.scribe = Scribe(self.quest_nr, self.station_nr, self.map_size ** 2)
        return self.get_state_object()

    def get_state_object(self):
        temp_map = copy.deepcopy(self.map)
        temp_map[self.player_pos[0]][self.player_pos[1]] = self.player_code

        state = np.reshape(temp_map, self.width * self.height)
        state = state / 255
        state = np.append(state, [self.gas / self.gas_max, np.clip(self.money / 500, 0, 1)])

        return state

    def step(self, action):
        if action > len(self.action_space) - 1:
            raise Exception('InvalidAction', action)
        else:
            reward = self.actions[action]()
            if self.gas <= 0:
                reward -= self.death_reward
                # reward -= self.reward_normalizer * 0.1
                self.is_done = True
                self.scribe.died[self.player_pos[0] + self.player_pos[1] * 15] += 1

        additional_reward, change_dict = self.action_special()
        reward += additional_reward
        if self.normalize_reward:
            reward /= self.reward_normalizer
        state = (self.get_state_object(), reward, self.is_done)
        return state

    def action_up(self):
        if self.player_pos[0] == 0:
            return self.action_wait()
        else:
            self.player_pos[0] -= 1

            self.gas -= 1
            return - 1 * self.gas_price

    def action_down(self):
        if self.player_pos[0] == self.height - 1:
            return self.action_wait()
        else:
            self.gas -= 1
            self.player_pos[0] += 1
            return - 1 * self.gas_price

    def action_right(self):
        if self.player_pos[1] == self.width - 1:
            return self.action_wait()
        else:
            self.gas -= 1
            self.player_pos[1] += 1
            return - 1 * self.gas_price

    def action_left(self):
        if self.player_pos[1] == 0:
            return self.action_wait()
        else:
            self.gas -= 1
            self.player_pos[1] -= 1
            return - 1 * self.gas_price

    def action_wait(self):
        self.scribe.invalid_action += 1
        self.gas -= 1 * self.wait_discount
        return -1 * self.wait_discount * self.gas_price

    def action_special(self):
        tile_code = self.map[self.player_pos[0]][self.player_pos[1]]
        if tile_code == 0:
            return 0, None
        elif tile_code == self.station_code:
            cost = self.gas - self.gas_max
            self.has_tanked = True
            self.scribe.tanked += 1
            if abs(cost) > self.money:
                cost = self.money * -1
                self.money = 0
                self.gas -= cost
            else:
                self.money += cost
                self.gas = self.gas_max
            # print("Tanking. Cost: ", cost)
            return 0, None
        elif self.quest_code <= tile_code < (self.quest_code + self.quest_nr):
            quest = int(tile_code - self.quest_code)
            if self.cargo[quest] == 0:
                self.scribe.picked[quest] += 1
                self.cargo[quest] = 1
                change_dict = self.start_quest(quest)
                self.money += self.rewards[quest] * self.prepaid

                return self.quest_reward * self.prepaid * self.quest_multiplier, change_dict
        elif self.reward_code <= tile_code < (self.reward_code + self.quest_nr):
            quest = int(tile_code - self.reward_code)
            if self.cargo[quest] == 1:
                reward, change_dict = self.end_quest(quest)
                self.money += reward
                # return reward * self.quest_multiplier * (1 - self.prepaid), False
                return self.quest_reward * self.quest_multiplier * (1 - self.prepaid), change_dict
                # return 10
        return 0, None

    def start_quest(self, quest_index):
        pos = self.player_pos
        destination = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
        while self.map[destination[0]][destination[1]] != 0:
            destination = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
        self.map[destination[0]][destination[1]] = self.reward_code + quest_index
        self.map[pos[0]][pos[1]] = 0
        manhattan_distance = abs(pos[0] - destination[0]) + abs(pos[1] - destination[1])
        reward = self.reward_per_step * (manhattan_distance ** self.large_quest_bonus)
        # reward *= random.random(self.random_deviation) - random.random(self.random_deviation * 2)
        reward += reward * random.uniform(-self.random_deviation, self.random_deviation)
        change_dict = {"type": 1, "index": quest_index, "x": destination[0], "y": destination[1], "reward": reward}
        self.rewards[quest_index] = reward
        return change_dict

    def end_quest(self, quest_index):
        self.scribe.ended[quest_index] += 1
        self.cargo[quest_index] = 0
        reward = (1 - self.prepaid) * self.rewards[quest_index]
        self.rewards[quest_index] = 0
        self.map[self.player_pos[0]][self.player_pos[1]] = 0
        destination = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
        while self.map[destination[0]][destination[1]] != 0:
            destination = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
        self.map[destination[0]][destination[1]] = self.quest_code + quest_index
        change_dict = {"type": 0, "index": quest_index, "x": destination[0], "y": destination[1], "reward": 0}
        # return 2
        return reward, change_dict


    def sample_move(self):
        return random.randint(0, len(self.action_space) - 1)

    def random_move(self):
        action = random.randint(0, len(self.action_space) - 1)
        _, _, is_done, should_oracle = self.step(action)
        return is_done, should_oracle

    def print_map(self, map=None):
        print("")
        print("Actions: 0: up, 1: down, 2: left, 3: right")
        print("")


        for i in range(self.map_size):
            line = "     "
            for j in range(self.map_size):
                if map is None:
                    line += self.map[i][j] + " "
                else:
                    line += "X" + str(trunc(map[i][j])) + " "
            print(line)

    def set_random_points(self, player_pos):
        points = [player_pos]
        dict = {}

        dict["player"] = {"x": player_pos[0], "y":player_pos[1]}


        if not self.uniform_gas_stations:
            for i in range(self.station_nr):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                while [x, y] in points:
                    x = random.randint(0, self.width - 1)
                    y = random.randint(0, self.height - 1)
                points.append([x, y])
                self.map[x][y] = self.station_code
        else:
            for i in range(self.station_nr):
                point = self.uniform_gas_list[i]
                x, y = point
                points.append([x, y])
                self.map[x][y] = self.station_code


        for i in range(self.quest_nr):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            while [x, y] in points:
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
            dict[i] = {"x": x, "y": y}
            points.append([x, y])
            self.map[x][y] = self.quest_code + i
        return dict


env = ConvEnv(quest_nr=4, station_nr=6, width=15, height=15, uniform_gas_stations=True, normalize_rewards=True)


env.step(0)




