from torchvision import transforms

class Dataset(object):
	"""docstring for Dataset"""
	def __init__(self):
		super(Dataset, self).__init__()
		self.data_dir = 'flower_data'
		self.train_dir = data_dir + '/train'
		self.valid_dir = data_dir + '/valid'

	def prepare(self):
		data_transforms = {
		    'train': transforms.Compose([
		        transforms.RandomRotation(30),
		        transforms.RandomResizedCrop(224),
		        transforms.RandomHorizontalFlip(),
		        transforms.ToTensor(),
		        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		    ]),
		    'valid': transforms.Compose([
		        transforms.Resize(256),
		        transforms.CenterCrop(224),
		        transforms.ToTensor(),
		        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		    ]),
		}

		image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), data_transforms[x]) for x in ['train', 'valid']}

		data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=0) for x in ['train', 'valid']}

		return data_loaders