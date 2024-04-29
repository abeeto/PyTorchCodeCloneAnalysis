import math, pygame

class smallFunctions:
    def absHeading(rad): #ONLY for display headings, dont fucking use for any calcs
        rad = rad + math.pi/2 # 90deg correction factor because 0 is to the right
        if rad > (2*math.pi):
            overshootFactor = math.floor(rad/(2*math.pi))
            rad = rad-(overshootFactor*2*math.pi)
        elif rad < 0:
            overshootFactor = math.floor(-rad/(2*math.pi))
            rad = rad+(overshootFactor*2*math.pi)
            rad = 2*math.pi + rad
        else:
            pass
        rad = round(math.degrees(rad))
        return str(rad)



    def rot_center(image, rect, angle):
        """rotate an image while keeping its center"""
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = rot_image.get_rect(center=rect.center)
        return rot_image,rot_rect