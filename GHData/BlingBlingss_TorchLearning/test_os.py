import os
path = "Sulog"
print("hello")
if not os.path.exists(path):
    print("NO, Create")
    os.makedirs(path)
print(os.path.join(path,"hparams"))