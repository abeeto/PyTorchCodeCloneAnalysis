import argparse

myarg = argparse.ArgumentParser()

myarg.add_argument("--data-path", default="helloworld",help="dataset path")
myarg.add_argument("--mode",default="FFA-net",type =str, help ="modelname")
myarg.add_argument("--device",default="cuda",help="device")


print(myarg)
opt = myarg.parse_args()
print(opt)