import pygame
import torch
import networks
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", help="The class name of the model to load (classes from networks.py)")
parser.add_argument("-f","--modelfile", default="", help="the filename of the class to load (if it has been changed)")
parser.add_argument("-d","--deviceid", default="", help="specify a device to use, eg: cpu or cuda:0")
parser.add_argument("-s","--scale", default="30", help="scale multiplier for the image (size of each pixel)")

args = parser.parse_args()
print("args",args)

if args.deviceid=="":
    if torch.cuda.is_available():
      device_id = "cuda"
    else:  
      device_id = "cpu"
else:
    device_id = args.deviceid
print("device id:",device_id)

device = torch.device(device_id)

image_size = 32
scale = int(args.scale)
text_size = int((image_size * scale)/11)

pygame.font.init()
FONT = pygame.font.SysFont('sans-serif', text_size)

text_split = FONT.get_height()

model_class = getattr(networks, args.model)
model = model_class()
if args.modelfile=="":
    model.load_state_dict(torch.load("models/"+args.model+".model"))
else:
    model.load_state_dict(torch.load("models/"+args.modelfile+".model"))
model.to(device)
model.eval()

screen = pygame.display.set_mode((image_size*scale + 500, image_size*scale))
pygame.init()
clock = pygame.time.Clock()

guesses = torch.zeros(10)
image = torch.zeros((image_size,image_size))

softmax = torch.nn.Softmax(1)

def draw_pixel(image,x,y,set_to):
    if x < image_size and x>=0 and y <image_size and y>=0:
        if set_to == 0 or image[int(x)][int(y)]<set_to:
            image[int(x)][int(y)] = set_to

def draw_to_image(set_to,prev_pos,pos):
    #y = pos[0]
    #x = pos[1]
    dist = int(((prev_pos[0]-pos[0])**2 + (prev_pos[1]-pos[1])**2) ** 0.5)
    for i in range(dist + 1):
        lerp = i/(dist + 1.0)
        y = (pos[0]-prev_pos[0])* lerp + prev_pos[0]
        x = (pos[1]-prev_pos[1])* lerp + prev_pos[1]
        draw_pixel(image,x/scale,y/scale,set_to)
        
        draw_pixel(image,x/scale,y/scale-1,set_to/2)
        draw_pixel(image,x/scale,y/scale+1,set_to/2)
        draw_pixel(image,x/scale+1,y/scale,set_to/2)
        draw_pixel(image,x/scale-1,y/scale,set_to/2)
        
        draw_pixel(image,x/scale-1,y/scale-1,set_to/3)
        draw_pixel(image,x/scale-1,y/scale+1,set_to/3)
        draw_pixel(image,x/scale+1,y/scale-1,set_to/3)
        draw_pixel(image,x/scale+1,y/scale+1,set_to/3)
        
        draw_pixel(image,x/scale,y/scale-2,set_to/3)
        draw_pixel(image,x/scale,y/scale+2,set_to/3)
        draw_pixel(image,x/scale+2,y/scale,set_to/3)
        draw_pixel(image,x/scale-2,y/scale,set_to/3)
    guesses[:] = softmax(model(image.reshape(1,1,image_size,image_size).to(device)))
    #guesses[:] = model(image.reshape(1,1,image_size,image_size))
        

running = True
mouse_pos = pygame.mouse.get_pos()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
            
    mouse_pressed = pygame.mouse.get_pressed()
    
    mouse_prev_pos = mouse_pos
    mouse_pos = pygame.mouse.get_pos()
    #print(mouse_pressed)
    if mouse_pressed[0]==1:
        draw_to_image(1,mouse_prev_pos,mouse_pos)
    elif mouse_pressed[2] == 1:
        draw_to_image(0,mouse_prev_pos,mouse_pos)
    
    key = pygame.key.get_pressed()
    if key[pygame.K_SPACE]:
        for i in range(image_size):
            image = torch.zeros((image_size,image_size))
            guesses = torch.zeros(10)
    
    screen.fill([255, 255, 255])
    #print(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            val = min(255,max(0,image[j][i]*255.0))
            pygame.draw.rect(screen,(val,val,val),
                (i*scale,j*scale,scale,scale))
    for i in range(guesses.shape[0]):
        text_surface = FONT.render(str(i)+': {:.1f}%'.format(guesses[i]*100), False, (0, guesses[i]*255, 0))
        screen.blit(text_surface,((image_size*scale + 30,14+(i+1)*text_split)))
            
    text_surface = FONT.render("guess: "+str(int(torch.argmax(guesses))), False, (0, 0, 0))
    screen.blit(text_surface,((image_size*scale + 30,10)))
    
    pygame.display.update()
    clock.tick(30)