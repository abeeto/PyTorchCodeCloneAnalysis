import torch
import argparse
import torchvision
from torchvision.models.alexnet import model_urls as alexnet_url
from torchvision.models.vgg import model_urls as vgg_url
from torchvision.models.resnet import model_urls as resnet_url

models = ['alexnet', 'vgg11', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

def select_model(model_name):
	if model_name in models:
		if model_name == models[0]:
			alexnet_url[model_name] = alexnet_url[model_name].replace('https://', 'http://')
			model = torchvision.models.alexnet(pretrained=True)
		elif model_name == models[1]:	
			vgg_url[model_name] = vgg_url[model_name].replace('https://', 'http://')
			model = torchvision.models.vgg11(pretrained=True)
		elif model_name == models[2]:
			vgg_url[model_name] = vgg_url[model_name].replace('https://', 'http://')	
			model = torchvision.models.vgg16(pretrained=True)
		elif model_name == models[3]:
			vgg_url[model_name] = vgg_url[model_name].replace('https://', 'http://')
			model = torchvision.models.vgg19(pretrained=True)
		elif model_name == models[4]:
			resnet_url[model_name] = resnet_url[model_name].replace('https://', 'http://')	
			model = torchvision.models.resnet18(pretrained=True)
		elif model_name == models[5]:
			resnet_url[model_name] = resnet_url[model_name].replace('https://', 'http://')
			model = torchvision.models.resnet34(pretrained=True)
		elif model_name == models[6]:
			resnet_url[model_name] = resnet_url[model_name].replace('https://', 'http://')
			model = torchvision.models.resnet50(pretrained=True)
		elif model_name == models[7]:
			resnet_url[model_name] = resnet_url[model_name].replace('https://', 'http://')
			model = torchvision.models.resnet101(pretrained=True)
		else:
			resnet_url[model_name] = resnet_url[model_name].replace('https://', 'http://')
			model = torchvision.models.resnet152(pretrained=True)
	
	else:
		print('input error. type python download_pretrained --help')	


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--download', default='vgg16', help='alexnet, vgg11, vgg16, vgg19, resnet18, resnet34, resnet50, resnet101, resnet152')
	args = parser.parse_args()
	
	arg = vars(args)
	select_model(arg['download'])	

