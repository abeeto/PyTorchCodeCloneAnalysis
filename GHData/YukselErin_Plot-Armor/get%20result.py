import runpy
#I use this to play a set amount of games and save their results as files, in order to use them as data for training
for i in range(100):
    runpy.run_path(path_name='game1.py ', run_name= str(i))