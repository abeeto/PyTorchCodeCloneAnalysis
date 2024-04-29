from os import system as shell

print("Setup The Envirenement Please Wait ...\n")

try:
	shell("pip install -r requirements.txt")
except Exception as e:
	print("You probably have\'t python installed!\n", e)
else:
	pass
finally:
	print("You probably have\'t python installed!\n")


print("Setup Finished - You Can Begin The Training Process!\n")
print("Starting The Training For You ...\n")
shell('python train.py')
print("Training Completed!\n")

print("To Predict a Flower Type Run The Command Bellow:\n")
print("python predict.py <path to the flower image>\n")

print("Good Luck Dude - Enjoy Deep Learning ^^")