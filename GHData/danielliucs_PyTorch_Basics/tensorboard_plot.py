import math
from tensorboardX import SummaryWriter


if __name__ == "__main__":
    writer = SummaryWriter()

    funcs = {"sin": math.sin, "cos": math.cos, "tan": math.tan} #defines a dictionary with key mapped to math operation

    for angle in range(-360, 360): #loop through angles
        angle_rad = angle * math.pi / 180 #convert to rads
        for name, fun in funcs.items(): #name is key, fun is math operation
            val = fun(angle_rad) #math operation(angle)
            writer.add_scalar(name, val, angle) #adding angles over and over again to form graph

    writer.close()

    
    #tensorboard --logdir runs --host localhost in terminal to see plot
