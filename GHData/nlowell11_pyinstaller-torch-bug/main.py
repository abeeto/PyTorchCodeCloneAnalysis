#test load
import time

if __name__ == '__main__':
	try:
		from fastai.vision import *
		
		defaults.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		print(defaults.device)  #Should print either cuda or cpu
		learn = load_learner('.','test_model.pkl')

		test_img = open_image("elephant.jpg")
		pred = learn.predict(test_img)

		if str(pred[0]) == "Elephants":
			print("Test classification SUCCESSFUL")
	finally:
		print("waiting 1s to close")
		time.sleep(1)

