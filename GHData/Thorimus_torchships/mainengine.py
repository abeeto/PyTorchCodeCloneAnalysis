from smallfunctions import smallFunctions
import pygame, sys, math, os
pygame.font.init()


# UNITS:
# 1 distance unit = 1km, by default 1 pixel
# 1 km/s = 1/60 "velocity units"
# due to this a velocity of 1 means a vel of 3 pixels per tick*TIMECONSTANT, or 1 km per second. Same for accel.
# 1 mass unit = 1 ton
# big G is just plain fucked


class vesselType:
    #Vessel type created from .txt at startup
    def __init__(self, typeName, maxThrust, fuelTons, waterTons, missileType = None, magazine = 0, emptyMass = 1, texture = 'ship.png', launchDirection = 0, biography = " "):
        self.typeName = str(typeName) #name of ship class for display
        self.maxThrust = maxThrust #max thrust
        self.waterTons = waterTons
        self.fuelTons = fuelTons
        self.missileType = missileType #type of missile loaded onto vessel
        self.magazine = magazine #no. of missiles loaded onto vessel
        self.emptyMass = emptyMass #total mass
        img_path = os.path.join(os.path.dirname(__file__), "img", texture)
        self.texture = pygame.image.load(img_path).convert_alpha() #sprite png
        self.launchDirection = launchDirection #direction of missile launch
        self.biography = biography
    def __str__(self) -> str:
        return self.typeName

class vessel:
    #Individual vessel created from vesselType object given in shipClass at game start
    def __init__(self, shipClass, name, isPlayer, x, y, velocity = [0, 0], pointing = 0, currentThrust = 0, missileType = None, missilesLoaded = None):

        #Dynamic vars
        self.x = x #pos x
        self.y = y #pos y
        self.velocity = list(velocity) #velocity vector
        self.pointing = pointing #where ship is pointing. 0 is right because of the trig functions and the way the coords are.
        self.currentThrust = currentThrust #thrust setting
        self.fixPoint = None
        self.target = None
        self.guidedMissiles = [] #missiles currently being guided by ownship

        #Static vars
        self.isPlayer = isPlayer
        self.shipClass = shipClass
        self.givenName = name
        self.id = len(activeShips)+1

        #If no missile specified, use default missile type of shipclass
        if missileType == None:
            self.missileType = shipClass.missileType 
        else:
            self.missileType = missileType

        #If specified missile load is higher than shipclass.magazine or unspecified then magazine is set to shipclass.magazine. If dumb shit is entered you get no missiles.
        if missilesLoaded == None or (int(missilesLoaded) > self.shipClass.magazine):
            self.magazine = self.shipClass.magazine
        elif int(missilesLoaded) <= self.shipClass.magazine:
            self.magazine = int(missilesLoaded)
        else:
            self.magazine = 0
    
        self.maxThrust = int(shipClass.maxThrust) #max thrust
        self.emptyMass = int(shipClass.emptyMass)
        self.waterTons = float(shipClass.waterTons)
        self.fuelTons = float(shipClass.fuelTons)
        self.mass = self.emptyMass + self.waterTons + self.fuelTons + (self.magazine * self.missileType.mass) #total mass
        self.texture = self.shipClass.texture #sprite png
        self.launchDirection = self.shipClass.launchDirection #direction of missile launch

        self.isTerminated = False

    def __str__(self):
        a = str(self.shipClass) + '_' + str(self.givenName)
        return a
        
    def update(self):
        
        # Update mass:
        self.mass = self.emptyMass + self.waterTons + self.fuelTons + (self.magazine * self.missileType.mass)

        # Limit thrust if no water or fuel:
        if self.waterTons == 0:
            if self.currentThrust > 0.3:
                self.currentThrust = 0.3
        
        if self.fuelTons <= 0:
            self.currentThrust = 0


        # Set thrustForce - basic equation that can be replaced with realistic rocket equiationg later:
        thrustForce = self.currentThrust*self.maxThrust*TIMEFACTOR
        
        # Use water to boost thrust:
        if self.currentThrust > 0.3:
            self.waterTons -= thrustForce * 1/700 # about 1 second of max thrust per ton of water

        # Use Fuel:
        self.fuelTons -= thrustForce * 10**-4 # about 9 minutes at full power


        # GRAVITY:
        for planet in gravObjects:
            dx = planet.x - self.x
            dy = planet.y - self.y
            angle = math.atan2(dy, dx)  # Calculate angle between planets
            d = math.sqrt((dx ** 2) + (dy ** 2))  # Calculate distance
            if d == 0:
                d = 0.000001  # Prevent division by zero error
            f = (
                G * self.mass * planet.mass / (d ** 2)
            )  # Calculate gravitational force

            if d < planet.radius:
                self.terminate()

            self.velocity[0] += (math.cos(angle) * f) / self.mass #iterate velocity based on each other planets force
            self.velocity[1] += (math.sin(angle) * f) / self.mass

        #iterate velocity based on thrustForce:
        self.velocity[0] += (math.cos(self.pointing) * thrustForce/self.mass)
        self.velocity[1] += (math.sin(self.pointing) * thrustForce/self.mass)

        #iterate pos based on velocity:
        self.x += self.velocity[0]
        self.y += self.velocity[1]

    def launchMissile(self):
        if self.isPlayer == True:
            if self.magazine > 0:
                x = firedMissile(self.missileType, len(activeMissiles)+1, self, self.target, self.fixPoint, True)
                self.guidedMissiles.append(x)
                activeMissiles.append(x)
                self.fixPoint = None
                self.magazine -= 1
            else:
                pass
        else:
            x = firedMissile(self.missileType, (len(activeMissiles)+1), self) ##WARNING CHANGE ID FOR AI
            self.guidedMissiles.append(x)
            activeMissiles.append(x)
            self.magazine -=1

    def blitta(self):
        oldRect = self.texture.get_rect(center=(self.x-displayOffset[0],self.y-displayOffset[1]))
        shipImg, newRect = smallFunctions.rot_center(self.texture,oldRect,-math.degrees(self.pointing + (0.5 * math.pi)))
        gameDisplay.blit(shipImg, newRect)

    def terminate(self):
        if self.isTerminated == False:
            for ship in activeShips:
                if ship.id == self.id:
                    activeShips.remove(ship)






class planetObj:
    def __init__(self, id, x, y, velocity, mass, radius, texture):
        self.id = id
        self.x = x #start X-position
        self.y = y #start Y-position
        self.velocity = velocity #start velocity
        self.mass = mass #mass
        self.radius = radius #radius incl atmosphere in pixels, both physically and graphically. make sure it scales well.
        img_path = os.path.join(os.path.dirname(__file__), "img", texture)
        importedTexture = pygame.image.load(img_path).convert_alpha()
        self.texture = pygame.transform.scale(importedTexture, (radius*2, radius*2))

    def update(self):
        for planet in activePlanets:
            if planet.id != self.id:
                dx = planet.x - self.x
                dy = planet.y - self.y
                angle = math.atan2(dy, dx)  # Calculate angle between planets
                d = math.sqrt((dx ** 2) + (dy ** 2))  # Calculate distance
                if d == 0:
                    d = 0.000001  # Prevent division by zero error
                f = (
                    G * self.mass * planet.mass / (d ** 2)
                )  # Calculate gravitational force

                self.velocity[0] += (math.cos(angle) * f) / self.mass #iterate velocity based on each other planets force
                self.velocity[1] += (math.sin(angle) * f) / self.mass

        #iterate pos based on velocity:
        self.x += self.velocity[0]
        self.y += self.velocity[1]
    
    def blitta(self):
        gameDisplay.blit(self.texture, (self.x-displayOffset[0]-self.radius, self.y-displayOffset[1]-self.radius))








    
class satelliteObj:
    def __init__(self, id, orbitParent, orbitRadius, mass, radius, texture):
        self.id = id #string id, ex. 'earth'
        self.orbitParent = orbitParent
        self.orbitRadius = orbitRadius
        self.mass = mass #mass
        self.radius = radius #radius incl atmosphere in pixels, both physically and graphically. make sure it scales well.
        
        self.x = self.orbitParent.x + orbitRadius
        self.y = self.orbitParent.y
        self.velocity = [0 + self.orbitParent.velocity[0], (math.sqrt((G*self.orbitParent.mass)/self.orbitRadius)) + self.orbitParent.velocity[1]] #WARNING ADD PARENT VELOCITY

        img_path = os.path.join(os.path.dirname(__file__),"img", texture)
        importedTexture = pygame.image.load(img_path).convert_alpha()
        self.texture = pygame.transform.scale(importedTexture, (radius*2, radius*2))

    def update(self):
        parentObject = self.orbitParent
        dx = parentObject.x - self.x
        dy = parentObject.y - self.y
        angle = math.atan2(dy, dx)  # Calculate angle between planets
        d = math.sqrt((dx ** 2) + (dy ** 2))  # Calculate distance
        if d == 0:
            d = 0.000001  # Prevent division by zero error
        f = (
            G * self.mass * parentObject.mass / (d ** 2)
        )  # Calculate gravitational force

        self.velocity[0] += (math.cos(angle) * f) / self.mass #iterate velocity based on each other planets force
        self.velocity[1] += (math.sin(angle) * f) / self.mass

        #iterate pos based on velocity:
        self.x += self.velocity[0]
        self.y += self.velocity[1]
    
    def blitta(self):
        gameDisplay.blit(self.texture, (self.x-displayOffset[0]-self.radius, self.y-displayOffset[1]-self.radius))




class guidedMissile:
    def __init__(self, maxThrust, burnTime, mass, launchTexture, warhead, PNparam = 1, texture = 'wpn.png'):
        self.maxThrust = maxThrust #maximum thrust coefficient - not actually a thrust force, but an acceleration in pixels per tick
        self.mass = mass
        self.launchTexture = launchTexture
        self.burnTime = burnTime
        self.warhead = warhead
        self.PNparam = PNparam
        img_path = os.path.join(os.path.dirname(__file__), "img", texture)
        self.texture = pygame.image.load(img_path).convert_alpha()

class firedMissile:
    def __init__(self, type, id, launchVehicle, target, fixPoint = None, isPlayerMissile = False):
        self.type = type
        self.mass = type.mass
        self.maxThrust = type.maxThrust #maximum thrust coefficient - not actually a thrust force, but an acceleration in pixels per tick
        self.launchTexture = type.launchTexture
        self.texture = type.texture
        self.burnTime = type.burnTime
        self.warheadYield = type.warhead
        self.PNparam = type.PNparam

        self.id = id #id of individually launched missile
        self.launchVehicle = launchVehicle

        self.commandedThrust = 0.1
        self.x = launchVehicle.x
        self.y = launchVehicle.y
        self.velocity = [launchVehicle.velocity[0], launchVehicle.velocity[1]]
        self.target = target
        self.fixPoint = fixPoint
        self.pointing = launchVehicle.launchDirection

    def terminate(self):
        for missile in activeMissiles:
            if missile.id == self.id:
                activeMissiles.remove(missile)
                self.launchVehicle.guidedMissiles.remove(missile)

    def update(self):
        for planet in gravObjects:
            dx = planet.x - self.x
            dy = planet.y - self.y
            angle = math.atan2(dy, dx)  # Calculate angle between planets
            d = math.sqrt((dx ** 2) + (dy ** 2))  # Calculate distance
            if d < planet.radius:
                self.terminate()  # Collision with object
            f = (
                G * self.mass * planet.mass / (d ** 2)
            )  # Calculate gravitational force

            self.velocity[0] += (math.cos(angle) * f) / self.mass #iterate velocity based on each other planets force
            self.velocity[1] += (math.sin(angle) * f) / self.mass

        #iterate velocity based on currentThrust(0-1) times maxthrust:
        if self.burnTime > 0:
            self.velocity[0] += (math.cos(self.pointing) * ((self.commandedThrust*self.maxThrust)/self.mass))
            self.velocity[1] += (math.sin(self.pointing) * ((self.commandedThrust*self.maxThrust)/self.mass))

        #WARNING fuel burn formula - change later:
        self.burnTime = self.burnTime - self.commandedThrust

        #iterate pos based on velocity:
        self.x += self.velocity[0]
        self.y += self.velocity[1]

    def missileGuidance(self): # ran continuosly

        if self.fixPoint != None:
            TGTx = self.fixPoint[0]
            TGTy = self.fixPoint[1]
            TGTxVel = 0
            TGTyVel = 0
            if abs(TGTx-self.x)<30 and abs(TGTy-self.y)<30: # Remove fixP if it has been reached
                self.fixPoint = None
        else:
            TGTx = self.target.x
            TGTy = self.target.y
            TGTxVel = self.target.velocity[0]
            TGTyVel = self.target.velocity[1]

        N = self.PNparam

        # Calculate relative velocity
        relVX = self.velocity[0] - TGTxVel
        relVY = self.velocity[1] - TGTyVel
        relVtot = math.sqrt(relVY**2+relVX**2)

        # Calculate range to TGT
        dx = self.x - TGTx
        dy = self.y - TGTy
        rangeTGT = math.sqrt(dx**2 + dy**2)
        # print(str(rangeTGT))
        
        # Calculate lead point
        Px = TGTx + N*rangeTGT*TGTxVel
        Py = TGTy + N*rangeTGT*TGTyVel
        # print(str(Px) + '    ' + str(Py))

        # Calculate commanded direction towards P
        commandedDirection = math.atan2((Py-self.y), (Px-self.x))
        #print(str(commandedDirection))

        self.pointing = commandedDirection
        self.commandedThrust = 0.02
        # self.commandedThrust=(
        #     ((-1)/(3*relVtot+1.5))+1
        # )

        # Prox fusing
        if rangeTGT < self.warheadYield:
            self.target.terminate()
            self.terminate()

    def blitta(self):
        oldRect = self.texture.get_rect(center=(self.x-displayOffset[0],self.y-displayOffset[1]))
        shipImg, newRect = smallFunctions.rot_center(self.texture,oldRect,-math.degrees(self.pointing + (0.5 * math.pi)))
        gameDisplay.blit(shipImg, newRect)



class player:
    def __init__(self):
        self.selectedFCS = playerShip #ownship by default but can be missiles in flight
        self.selectedTarget = None








class GUI:
    def __init__(self):
        img_path = os.path.join(os.path.dirname(__file__), 'img', 'fixPoint.png')
        self.GUIfixPoint = pygame.image.load(img_path).convert_alpha()
        self.GUIfixPointXY = [16,16]
        img_path = os.path.join(os.path.dirname(__file__), 'img', 'target.png')
        self.GUItarget = pygame.image.load(img_path).convert_alpha()
        self.GUItargetXY = [16,16]

    def blit(self):
        if playerShip.fixPoint != None:
            gameDisplay.blit(self.GUIfixPoint, (playerShip.fixPoint[0]-self.GUIfixPointXY[0], playerShip.fixPoint[1]-self.GUIfixPointXY[1]))
        if playerShip.target != None:
            gameDisplay.blit(self.GUItarget, (playerShip.target.x-displayOffset[0]-self.GUItargetXY[0], playerShip.target.y-displayOffset[1] - self.GUItargetXY[1]))





def draw():
    gameDisplay.blit(spaceBackground, (0,0))

    mx, my = pygame.mouse.get_pos()

    # TELEMETRY
    myfont = pygame.font.SysFont('TratexSvart', 30)
    absVel = math.sqrt(playerShip.velocity[0]**2 + playerShip.velocity[1]**2)
    #moonVel = math.sqrt(moon.velocity[0]**2 + moon.velocity[1]**2)
    telemetry = (
        'THR:' + str(round(playerShip.currentThrust, 3)) +
        ' VEL: ' + str(round(absVel, 3)) +
        ' HDG: ' + smallFunctions.absHeading(playerShip.pointing) +
        ' HDGRAW: ' + str(round(playerShip.pointing, 2)) +
        ' TGT: ' + str(playerShip.target) + 
        ' MousePos: ' + str(mx + displayOffset[0]) + ' ' + str(my+displayOffset[1]) + 
        ' playerpos: ' + str(playerShip.x) + ' ' + str(playerShip.y) +
        ' selected missile: ' + str(playerShip)
        )
    textsurface = myfont.render(telemetry, False, (255, 0, 0))
    gameDisplay.blit(textsurface,(display_width * 0.18, display_height * 0.2))

    # PLANETS & SPACECRAFT
    for planet in activePlanets:
        planet.blitta() #blit to gamedisplay
    for ship in activeShips:
        ship.blitta() #blit to gamedisplay
    for satellite in activeSatellites:
        satellite.blitta() #blit to gamedisplay
    for missile in activeMissiles:
        missile.blitta()

    mainGUI.blit()
        
    pygame.display.update()




def keypresses():
    mx, my = pygame.mouse.get_pos()
    trueX = mx + displayOffset[0]
    trueY = my + displayOffset[1]

    events = pygame.event.get()
    keys = pygame.key.get_pressed()

  # proceed events
    for event in events:

        if event.type == pygame.QUIT: #input to close
            pygame.quit()
            sys.exit()

        # Handle MOUSEBUTTONUP
        if event.type == pygame.MOUSEBUTTONUP:
            print('click')
            # Targeting
            isShipTarget = False
            for ship in activeShips:
                if ship.x-20 < trueX < ship.x+20 and ship.y-20 < trueY < ship.y+20:
                    playerShip.target = ship
                    isShipTarget = True
                    
            if isShipTarget == False:
                player.selectedFCS.fixPoint = [trueX, trueY]
        
        # Handle instantaneous keypresses
        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_SPACE:
                if playerShip.target:
                    playerShip.launchMissile()
                    print('launched')

            elif event.key == pygame.K_x:
                playerShip.fixPoint = None


# Thrust & Movement
    if keys[pygame.K_LEFT]:
        playerShip.pointing += -0.1
    elif keys[pygame.K_RIGHT]:
        playerShip.pointing += 0.1
    if keys[pygame.K_UP]:
        if playerShip.currentThrust < 1:
            playerShip.currentThrust += 0.01
    elif keys[pygame.K_DOWN]:
        if playerShip.currentThrust >= 0:
            playerShip.currentThrust += -0.01
    if playerShip.currentThrust < 0:
        playerShip.currentThrust = 0
    
# View
    if keys[pygame.K_w]:
        displayOffset[1] += -3
    elif keys[pygame.K_s]:
        displayOffset[1] += 3
    elif keys[pygame.K_a]:
        displayOffset[0] += -3
    elif keys[pygame.K_d]:
        displayOffset[0] += 3





# INITIALISATIONS
pygame.init()

display_width = 1920
display_height = 1080

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('space game')
clock = pygame.time.Clock()



spaceBackground = pygame.image.load(os.path.join("img", 'skybox.jpg'))



# CONSTANTS AND OBJECTS
G = 0.3
#6.67408 * (10 ** -11) #main grav constant

activeShips = [] # Affected by all gravOjects = planets + satellites
activeStations = [] # Only affected by parent body

activePlanets = [] # Only affected by other planets
activeSatellites = [] # Only affected by parent body

activeMissiles = []
playerMissiles = []


displayOffset = [0,0]
TIMEFACTOR = 1/60


mainGUI = GUI()



""" TEMP PLANET AND SHIP CREATION """
K700 = guidedMissile(27, 300, 27, 'bullshit.png', 20, 1)

testFrigate = vesselType('Test Frigate', 670, 30, 57, K700, 40, 250, 'ship.png', 0)

playerShip = vessel(testFrigate, 'shandong', True, 200, 200, [0.1,0], 0, 1)
activeShips.append(playerShip)

othership = vessel(testFrigate, 'seconddong', False, 400, 400, [0.1,0], 0.2)
activeShips.append(othership)

# mars = planetObj('mars', display_width*0.6, display_height*0.5, [0, -1], 3000, 32, 'planet.png')
# activePlanets.append(mars)

# earth = planetObj('earth', display_width*0.4, display_height*0.5, [0, 0], 3000, 128, 'earthpng.png')
# activePlanets.append(earth)

# moon = satelliteObj('luna', earth, 300, 3000, 32, 'moon.png')
# activeSatellites.append(moon)

# deimos = satelliteObj('deimos', moon, 300, 3000, 32, 'moon.png')
# activeSatellites.append(deimos)


gravObjects = activePlanets + activeSatellites
""" END TEMP PLANET AND SHIP CREATION """







# MAIN LOOP
while True:
    clock.tick(60)

    mx, my = pygame.mouse.get_pos()

    keypresses()

    #physics iteration:
    for ship in activeShips:
        ship.update()
    for planet in activePlanets:
        planet.update()
    for sat in activeSatellites:
        sat.update()
    for missile in activeMissiles:
        missile.update()
        missile.missileGuidance()

    #draw shit:
    draw()

   
    


