# code from https://stackoverflow.com/questions/54003194/how-can-i-access-tablet-pen-data-via-python

import pyglet
import numpy as np
from pyglet.gl import *
from array_image import *

window = pyglet.window.Window()
tablets = pyglet.input.get_tablets()
canvases = []

scale = 10
#rectangle = pyglet.shapes.Rectangle(0,0,10,10)
image = np.zeros((32*scale,32*scale))
arr_img = ArrayImage(image)

if tablets:
    print('Tablets:')
    for i, tablet in enumerate(tablets):
        print('  (%d) %s' % (i + 1, tablet.name))
    print('Press number key to open corresponding tablet device.')
else:
    print('No tablets found.')
    


index = 0

if not (0 <= index < len(tablets)):
    prin("no tablet connected")
    exit()

name = tablets[i].name

try:
    canvas = tablets[i].open(window)
    print("canvas",type(canvas),canvas)
except pyglet.input.DeviceException:
    print('Failed to open tablet %d on window' % index)

print('Opened %s' % name)

@canvas.event
def on_enter(cursor):
    print('%s: on_enter(%r)' % (name, cursor))

@canvas.event
def on_leave(cursor):
    print('%s: on_leave(%r)' % (name, cursor))

@canvas.event
def on_motion(cursor, x, y, pressure, a, b):  # if you know what "a" and "b" are tell me (tilt?)
    print('%s: on_motion(%r, x=%r, y=%r, pressure=%r, %s, %s)' % (name, cursor, x, y, pressure, a, b))
    #print(type(canvas.window),canvas.window,canvas.window.get_location())
    image_x = x
    image_y = y+88
    print("image_x/y",image_x,image_y)
    if pressure>0.0:
        if 0<=image_x<32*scale and 0<=image_y<32*scale:
            print("drawed")
            for i in range(10):
                for j in range(10):
                    if (i-5)**2+(j-5)**2<=25:
                        #image[image_y+i][image_x+j] = int(pressure * 255)
                        image[image_y+i][image_x+j] = pressure
        window.invalid = True
    
@window.event
def on_draw():
    print("drawing")
    arr_img.update()
    arr_img.image.blit(0, 0)
    #for i in range(32):
    #    for j in range(32):
    #        rectangle._x = i*10
    #        rectangle._y = j*10
    #        rectangle._update_position()
    #        rectangle.color = (image[i][j],image[i][j],image[i][j])
    #        rectangle.draw()

@window.event
def on_text(text):
    image[:][:]=0

@window.event
def on_mouse_press(x, y, button, modifiers):
    print('on_mouse_press(%r, %r, %r, %r' % (x, y, button, modifiers))

@window.event
def on_mouse_release(x, y, button, modifiers):
    print('on_mouse_release(%r, %r, %r, %r' % (x, y, button, modifiers))

pyglet.app.run()