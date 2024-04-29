import time

class Car:


    """
    Triangulates each car and computes it's current speed
    """

    def getMPH(self, time, distance):
        self.fps = distance / time
        self.MPH = int(self.fps / 1.467)

        return self.MPH


    def __init__(self, xPos, yPos):

        self.xPos = xPos
        self.yPos = yPos

        # Locations of landmarks on screen in pixels
        self.pt1 = (520, 133)
        self.pt2 = (164, 178)
        self.pt3 = (69, 271)

        # Time when car passed a pt
        self.pt1Passed = 0
        self.pt2Passed = 0
        self.pt3Passed = 0

        self.speed = 0


    def locate(self, xPos, yPos):
        current_time = time.time();

        if len(xPos) > 3:
            self.fx = float(xPos[8:15])
            self.fy = float(yPos[8:15])
            if(abs(self.fx - self.pt1[0]) <= 25):
                if(abs(self.fy - self.pt1[1]) <= 25):
                    self.pt1Passed = current_time

            if(abs(self.fx - self.pt2[0]) <= 25):
                if(abs(self.fy - self.pt2[1]) <= 25):
                    self.pt2Passed = current_time
                    self.speed = str(self.getMPH((current_time - self.pt1Passed), 317))

    
    def getSpeed(self):
        return self.speed




