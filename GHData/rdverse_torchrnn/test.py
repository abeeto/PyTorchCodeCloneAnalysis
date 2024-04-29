import yaml
import os
print(os.listdir('config'))
configFile = open('config/model1.yaml').readlines()
conf = yaml(configFile)
print(type(conf))
print(conf)
