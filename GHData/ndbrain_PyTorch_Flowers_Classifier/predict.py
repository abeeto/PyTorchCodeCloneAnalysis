import sys
from model import Model
from PIL import Image
import json

class Predict(object):
	"""docstring for Predict"""
	def __init__(self, img=str(sys.argv[1])):
		super(Predict, self).__init__()
		self.img = img
		self.model = Model()
		self.checkpoint = torch.load('trained/checkpoint.cpt') # note: don't change the 'checkpoint.cpt' file name!
		self.model.load_state_dict(self.checkpoint)

		with open('cat_to_name.json', 'r') as f:
    	self.cat_to_name = json.load(f)

	def load_image(self, img=self.img, max_size=400, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
    
    image = Image.open(img).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    
    return image

	def run(self):
		cuda = torch.cuda.is_available()
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		image = self.load_image().to(device)
		model.to(device)
		output = model(image)
		# convert output probabilities to predicted class
		_, max_idx = torch.max(output, 1)
		pred = np.squeeze(preds_tensor.numpy()) if not cuda else np.squeeze(preds_tensor.cpu().numpy())

		print('The Flower Type is:', )

		