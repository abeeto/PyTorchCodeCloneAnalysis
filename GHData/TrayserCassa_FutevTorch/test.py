#!/usr/bin/python3
from time import sleep
from rpi_ws281x import *
from threading import Thread

class FutevTorchMode():
        NONE = -1
        BLINK_SHORT = 0
        BLINK_LONG = 1
        PULSE = 2
        RAINBOW = 3

class FutevTorch():
        def __init__(self, pin=18, frequency=800000, dma=5, brightness=255):
                self.__pin = pin
                self.__frequency = frequency
                self.__dma = dma
                self.__brightness = brightness

        def __init_torch(self):
                self.__torch = FutevTorchThread(self.__pin, self.__frequency, self.__dma, self.__brightness)

        def run_pulse(self, red, green, blue, intervall=0.005):
                self.__init_torch()
                self.__torch.set_color(red, green, blue)
                self.__torch.set_mode(FutevTorchMode.PULSE)
                self.__torch.set_intervall(intervall)
                self.__torch.start()

        def run_rainbow(self, intervall=0.02):
                self.__init_torch()
                self.__torch.set_intervall(intervall)
                self.__torch.set_mode(FutevTorchMode.RAINBOW)
                self.__torch.start()


        def run_blink_short(self, red, green, blue, intervall=3):
                self.__init_torch()
                self.__torch.set_color(red, green, blue)
                self.__torch.set_intervall(intervall)
                self.__torch.set_mode(FutevTorchMode.BLINK_SHORT)
                self.__torch.start()

        
        def run_blink_long(self, red, green, blue, intervall=3):
                self.__init_torch()
                self.__torch.set_color(red, green, blue)
                self.__torch.set_intervall(intervall)
                self.__torch.set_mode(FutevTorchMode.BLINK_LONG)
                self.__torch.start()

                
        def stop(self):
                self.__torch.stop()
                self.__torch.join()


class FutevTorchThread(Thread):
        def __init__(self, pin=18, frequency=800000, dma=5, brihtness=255):
                Thread.__init__(self)
                self.__strip = PixelStrip(1, pin, frequency, dma, False, brihtness)
                self.__strip.begin()

                self.__red = 0
                self.__green = 0
                self.__blue = 0

                self.__mode = FutevTorchMode.NONE
                self.__intervall = 0                

        def set_color(self, red, green, blue):
                self.__red = red
                self.__green = green
                self.__blue = blue


        def set_mode(self, mode):
                self.__mode = mode
                

        def set_intervall(self, intervall):
                self.__intervall = intervall
                

        def __show_pulse(self):
                while not self.__stop:
                        self.__smooth_start(self.__intervall)
                        self.__smooth_stop(self.__intervall)

        def __show_rainbow(self):
                self.set_color(0, 255, 0)
                self.__smooth_start(0.005)
                while not self.__stop:
                        for i in range(255):                                
                                red, green, blue = self.__wheel((i) & 255) 
                                self.set_color(red, green, blue)
                                self.__show()
                                sleep(self.__intervall)                
                self.__smooth_stop(0.005)


        def __wheel(self, position):
	        if position < 85:
		        return position * 3, 255 - position * 3, 0
	        elif position < 170:
		        position -= 85
		        return 255 - position * 3, 0, position * 3
	        else:
		        position -= 170
		        return 0, position * 3, 255 - position * 3
                

        def __show_blink(self, time):
                for i in range(self.__intervall):
                        self.__strip.setBrightness(255)
                        self.__show()
                        sleep(time)
                        self.__strip.setBrightness(0)
                        self.__show()
                        sleep(time)
                        
        
        def run(self):
                self.__stop = False

                if self.__mode == FutevTorchMode.PULSE:
                        self.__show_pulse()

                elif self.__mode == FutevTorchMode.RAINBOW:
                        self.__show_rainbow()

                elif self.__mode == FutevTorchMode.BLINK_SHORT:
                        self.__show_blink(0.5)

                elif self.__mode == FutevTorchMode.BLINK_LONG:
                        self.__show_blink(1)

                self.__strip._cleanup()
                        
        def __show(self):
                self.__strip.setPixelColorRGB(0, self.__red, self.__green, self.__blue)
                self.__strip.show()

        def stop(self):
                self.__stop = True

        def __smooth_start(self, intervall = 0.002):
                self.__strip.setPixelColorRGB(0, self.__red, self.__green, self.__blue)
                for i in range(0, 255, 1):
                        self.__strip.setBrightness(i)
                        self.__strip.show()
                        sleep(intervall)
                
        def __smooth_stop(self, intervall = 0.002):
                for i in range(255, 0, -1):
                        self.__strip.setBrightness(i)
                        self.__strip.show()
                        sleep(intervall)

                        
if __name__ == '__main__':
        torch = FutevTorch()
        #torch.run_pulse(255,0,255)
        #print("Pulse")
        #sleep(3)
        #torch.stop()

        torch.run_rainbow()
        print("Rainbow")
        sleep(30)
        torch.stop()

        torch.run_blink_short(255,0,0, intervall=10)
        print("blink short")
        sleep(3)
        torch.stop()

        torch.run_blink_long(0,0,255)
        print("blink short")
        sleep(3)
        torch.stop()
