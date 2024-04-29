from random import randint
import pygame
from numpy import random
import numpy as np


class Game(object):
    enemyTotalHealth = 70
    currentEnemyHealth = 70
    laserduration = 500
    width = 800
    height = 600

    class Player(object):
        health = 10
        coords = [400,590]
        left = True
        def move(self):
            if self.coords[0] == 790 or self.coords[0] == 10:
                self.left = not self.left
            if self.left:
                self.coords[0] -=1
            else:
                self.coords[0] +=1

    """"This object is used to keep track of laser shots that need to be drawn for multiple frames"""
    class Laser(object):
        start, finish, color, duration = (0,0),(0,0),(0,0,0),0
        def __init__(self,start, finish, color, duration ) :
            self.start, self.finish, self.color, self.duration = start, finish, color, duration 

    player = Player
    enemies = []
    gameDelay = 10
    defeated = 0
    shots = []
    enemyEvent = pygame.USEREVENT+1
    playerEvent= pygame.USEREVENT+2

    def __init__(self, agro, health, power):
        self.window = pygame.display.set_mode((self.width,self.height))
        pygame.time.set_timer(self.enemyEvent, int(2000/ agro))
        pygame.time.set_timer(self.playerEvent, 1000)
        self.agro = agro
        self.health = health
        self.power = power
    
    """This function draws a laser from the list and decrements its duration variable, which allows us to control for how many frames a line is drawn."""
    def drawShot(self,laser):
        pygame.draw.line(self.window, laser.color, laser.start, laser.finish)
        laser.duration -= 2
        return laser

    """"This function uses (shots) list to draw all the necessary lines on the screen for each frame."""
    def drawAllShots (self):
        for laser in self.shots:
            laser = self.drawShot(laser)
            if laser.duration <= 0:
                self.shots.remove(laser)

    """This function calculates coordinates for every enemy laser and puts them in a list. Also if the player is hit by the laser the player loses health."""
    def enemyActivity(self):
        for enemy in self.enemies:
            shotX = (enemy + random.randint(300) - 150)
            pygame.draw.line(self.window,(255,0,0), (enemy,20), (shotX, 600))
            laser = self.Laser((enemy,20), (shotX, 600), (255, 0,0), self.laserduration)
            self.shots.append(laser)
            if( 20 > abs(shotX - self.player.coords[0])):
                self.health -=20
                
    """"Creates a set number of enemy circles and keeps them in a list."""
    def createEnemy(self):
        gap = (self.width - 20) /5
        self.enemies.append(70)
        for i in range(1,5):
            self.enemies.append(i*gap + 70 )
    
    def drawEnemy(self):
        for enemy in self.enemies:
            pygame.draw.circle(self.window, (255,0,0), (enemy,20), 20)

    """The player shoots its laser at the closest target with perfect accuracy. Also by every shot the enemy loses health. Although there are multiple enemies represented in the visual they all share one health pool."""
    def playerActivity(self):
        min =801
        cord = -1
        for i in self.enemies:
            if abs(self.player.coords[0] -i )< min:
                min = abs(self.player.coords[0] -i )
                cord = i
        laser = self.Laser(self.player.coords, (cord,20), (0,255,0), self.laserduration)
        self.shots.append(laser)
        self.currentEnemyHealth -= self.power
        if self.currentEnemyHealth <= 0:
            self.enemies.pop()
            self.currentEnemyHealth = self.enemyTotalHealth


    def startGame(self):       
        self.window.fill((220,243,255))
        self.createEnemy()
        pygame.time.delay(self.gameDelay)
        gameover = False        
        while not gameover:
            self.player.move(self.player)
            self.window.fill((220,243,255))
            self.drawAllShots()
            pygame.draw.circle(self.window, (0,255,0),(self.player.coords),20)
            self.drawEnemy()
            for event in pygame.event.get():

                if event.type == self.enemyEvent:
                    self.enemyActivity() 
                if event.type == self.playerEvent:
                    self.playerActivity()
                    if not self.enemies:
                        return 1
                if (self.health <= 0):
                    return 0
            pygame.display.update()


#If the game is called from "prediction.py" it is given a set of parameters as comma seperated values on a string we use this parameters and print the result
if ',' in __name__:
    params = tuple(map(int, __name__.split(','))) 
    agro = int(params[0])
    health =  int(params[1])
    power =  int(params[2])
    game = Game(agro, health, power)
    result = game.startGame()
    if result == 1:
        print("Result is player victory!")
        pygame.quit()
    else:
        print("Result is enemy victory!")
        pygame.quit()
#We play a game with randomzied input parameters, and if a valid iteration number is given, the input paramaters and the output is saved in a .npy file
#I used this with "get result.py" to get a set of games to predict upon
else:
    agro, health, power = random.randint(9)+1, random.randint(160) +1, random.randint(46)+1
    cond = np.array([[agro, health, power]])
    txt = "iteration" + __name__ + ".npy"
    game = Game(agro, health, power)
    result = np.array([[0,0]])
    result[0][game.startGame()] = 1
    with open(txt, 'wb') as f:
        np.save(f,cond) 
        np.save(f, result)
    pygame.quit()

