import argparse
import os
import sys
import time

from  PIL import Image

from tqdm import tqdm

import os
import sys
import image_slicer

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import glob
import sys, shutil

import utils
from transformer_net import TransformerNet
from vgg import Vgg16


from math import floor


def check_paths(args):
	try:
		if not os.path.exists(args.save_model_path):
			os.makedirs(args.save_model_path)
		if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
			os.makedirs(args.checkpoint_model_dir)
	except OSError as e:
		print(e)
		sys.exit(1)

def resize(imge,maxsize):

	# img = Image.open(imge).convert('RGB')
	img = imge
	width, height = img.size
	# print("\n" + path+item)
	# print("\n\n",width, height)
	if width < height:
		new_height = maxsize
		new_width  = floor(new_height * width / height)
		# print(new_width,new_height)
		imResized = img.resize((new_width,new_height), Image.ANTIALIAS)
		return imResized
	elif width > height:
		new_width = maxsize
		new_height = floor(new_width * height / width)
		# print(new_width,new_height)
		imResized = img.resize((new_width,new_height), Image.ANTIALIAS)
		return imResized
	elif width == height:
		new_width = maxsize
		new_height = floor(new_width * height / width)
		# print(new_width,new_height)
		imResized = img.resize((new_width,new_height), Image.ANTIALIAS)
		return imResized
	# print(new_width,new_height)
	# f = out+item
	# imResized = img.resize((new_width,new_height), Image.ANTIALIAS)
	# return imResized
	# imResized.save(f + '_resized.jpg')

def train(args):
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	if args.cuda:
		torch.cuda.manual_seed(args.seed)

	transform = transforms.Compose([
		transforms.Resize(args.image_size),
		transforms.CenterCrop(args.image_size),
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x.mul(255))
	])
	train_dataset = datasets.ImageFolder(args.dataset, transform)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

	transformer = TransformerNet()
	optimizer = Adam(transformer.parameters(), args.lr)
	mse_loss = torch.nn.MSELoss()

	vgg = Vgg16(requires_grad=False)
	style_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x.mul(255))
	])
	style = utils.load_image(args.style_image, size=args.style_size)
	style = style_transform(style)
	style = style.repeat(args.batch_size, 1, 1, 1)

	if args.cuda:
		transformer.cuda()
		vgg.cuda()
		style = style.cuda()

	style_v = Variable(style)
	style_v = utils.normalize_batch(style_v)
	features_style = vgg(style_v)
	gram_style = [utils.gram_matrix(y) for y in features_style]

	for e in range(args.epochs):
		transformer.train()
		agg_content_loss = 0.
		agg_style_loss = 0.
		count = 0
		for batch_id, (x, _) in tqdm(enumerate(train_loader)):
			n_batch = len(x)
			count += n_batch
			optimizer.zero_grad()
			x = Variable(x)
			if args.cuda:
				x = x.cuda()

			y = transformer(x)

			y = utils.normalize_batch(y)
			x = utils.normalize_batch(x)

			features_y = vgg(y)
			features_x = vgg(x)

			content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

			style_loss = 0.
			for ft_y, gm_s in zip(features_y, gram_style):
				gm_y = utils.gram_matrix(ft_y)
				style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
			style_loss *= args.style_weight

			total_loss = content_loss + style_loss
			total_loss.backward()
			optimizer.step()

			agg_content_loss += content_loss.data[0]
			agg_style_loss += style_loss.data[0]

			if (batch_id + 1) % args.log_interval == 0:
				mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
					time.ctime(), e + 1, count, len(train_dataset),
								  agg_content_loss / (batch_id + 1),
								  agg_style_loss / (batch_id + 1),
								  (agg_content_loss + agg_style_loss) / (batch_id + 1)
				)
				print(mesg)

			if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
				transformer.eval()
				if args.cuda:
					transformer.cpu()
				ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
				ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
				# save_checkpoint(transformer.state_dict(),False,ckpt_model_path)
				torch.save(transformer.state_dict(), ckpt_model_path)
				if args.cuda:
					transformer.cuda()
				transformer.train()

	# save model
	transformer.eval()
	if args.cuda:
		transformer.cpu()
	save_model_filename = "final.pth"
	save_model_path = os.path.join(args.save_model_path, save_model_filename)
	torch.save(transformer.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)
	
	# np.random.seed(args.seed)
 #    torch.manual_seed(args.seed)

 #    if args.cuda:
 #        torch.cuda.manual_seed(args.seed)

 #    transform = transforms.Compose([
 #        transforms.Resize(args.image_size),
 #        transforms.CenterCrop(args.image_size),
 #        transforms.ToTensor(),
 #        transforms.Lambda(lambda x: x.mul(255))
 #    ])
 #    train_dataset = datasets.ImageFolder(args.dataset, transform)
 #    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

 #    transformer = TransformerNet()
 #    optimizer = Adam(transformer.parameters(), args.lr)
 #    mse_loss = torch.nn.MSELoss()

 #    vgg = Vgg16(requires_grad=False)
 #    style_transform = transforms.Compose([
 #        transforms.ToTensor(),
 #        transforms.Lambda(lambda x: x.mul(255))
 #    ])
 #    style = utils.load_image(args.style_image, size=args.style_size)
 #    style = style_transform(style)
 #    style = style.repeat(args.batch_size, 1, 1, 1)

 #    if args.cuda:
 #        transformer.cuda()
 #        vgg.cuda()
 #        style = style.cuda()

 #    style_v = Variable(style)
 #    style_v = utils.normalize_batch(style_v)
 #    features_style = vgg(style_v)
 #    gram_style = [utils.gram_matrix(y) for y in features_style]

 #    for e in range(args.epochs):
 #        transformer.train()
 #        agg_content_loss = 0.
 #        agg_style_loss = 0.
 #        count = 0
 #        for batch_id, (x, _) in enumerate(train_loader):
 #            n_batch = len(x)
 #            count += n_batch
 #            optimizer.zero_grad()
 #            x = Variable(x)
 #            if args.cuda:
 #                x = x.cuda()

 #            y = transformer(x)

 #            y = utils.normalize_batch(y)
 #            x = utils.normalize_batch(x)

 #            features_y = vgg(y)
 #            features_x = vgg(x)

 #            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

 #            style_loss = 0.
 #            for ft_y, gm_s in zip(features_y, gram_style):
 #                gm_y = utils.gram_matrix(ft_y)
 #                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
 #            style_loss *= args.style_weight

 #            total_loss = content_loss + style_loss
 #            total_loss.backward()
 #            optimizer.step()

 #            agg_content_loss += content_loss.data[0]
 #            agg_style_loss += style_loss.data[0]

 #            if (batch_id + 1) % args.log_interval == 0:
 #                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
 #                    time.ctime(), e + 1, count, len(train_dataset),
 #                                  agg_content_loss / (batch_id + 1),
 #                                  agg_style_loss / (batch_id + 1),
 #                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
 #                )
 #                print(mesg)

 #            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
 #                transformer.eval()
 #                if args.cuda:
 #                    transformer.cpu()
 #                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
 #                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
 #                torch.save(transformer.state_dict(), ckpt_model_path)
 #                if args.cuda:
 #                    transformer.cuda()
 #                transformer.train()



def transform_video(args):
	_in  = str(args.tmp + "/in/")
	_out = str(args.tmp + "/out/")	

	if not os.path.exists(args.tmp):
		os.makedirs(args.tmp)
		os.makedirs(_in)
		os.makedirs(_out)
	if os.path.exists(args.tmp):
		shutil.rmtree(args.tmp,ignore_errors=False, onerror=None)
		os.makedirs(args.tmp)
		os.makedirs(_in)
		os.makedirs(_out)

	if args.mode == 0:
		print("Folder Mode")
		if os.path.isdir(args.content_video):
			for filename in tqdm(glob.glob(args.content_video + '/*.png')):
				shutil.copy(filename, _in)
		else:
			print("ERROR! Your input was not a folder!")


	elif args.mode == 1:
		print("FFMPEG Mode")
		cmdl = 'ffmpeg -i ' + str(args.content_video) + ' -r ' + args.fps + '  '+ _in + '/frame%03d.png'
		os.system(str(cmdl))
		print("DONE!")
		# Put funtions here


	elif args.mode == 2:
		print("VideoSequence Mode")
		from contextlib import closing
		from videosequence import VideoSequence

		with closing(VideoSequence(args.content_video)) as frames:
			for idx, frame in enumerate(frames[:]):
				frame.save(_in + "frame{:04d}.png".format(idx))

	for idx, i in tqdm(enumerate(glob.glob(_in + '*.png'))):

		# print("input: " + str(i))
		savdir = _out + "/" + os.path.basename(i)
		sys.stdout.write(savdir)
		sys.stdout.flush()
		
		# print(savdir)

		content_image = utils.load_image(i, scale=args.content_scale)
		if args.max_size is not None:
			content_image = resize(content_image, args.max_size)
		content_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Lambda(lambda x: x.mul(255))
		])
		content_image = content_transform(content_image)
		content_image = content_image.unsqueeze(0)
		if args.cuda:
			content_image = content_image.cuda()
		content_image = Variable(content_image, volatile=True)

		style_model = TransformerNet()
		style_model.load_state_dict(torch.load(args.model))
		if args.cuda:
			style_model.cuda()

		output0 = style_model(content_image)
		output = style_model(output0)
		if args.cuda:
			output = output.cpu()
		output_data = output.data[0]

		utils.save_image(savdir, output_data)

	if args.mode == 0:
		print("Folder Mode")
		for filename in tqdm(glob.glob(_out)):
			shutil.copy(filename, args.output_video)
	elif args.mode == 1:
		print("FFMPEG Mode. Stitch it back together.")
		cmdl = 'ffmpeg -framerate ' + args.fps + ' -i ' + str(_out + '/frame%03d.png   ') + args.output_video
		os.system(str(cmdl))
		print("DONE!")
	elif args.mode == 2:
		print("VideoSequence Mode. Stitch it back together.")


def stylize(args):
	if os.path.isdir(args.model):
		# processing_ALL_MODELS!!!
		for modX in tqdm(glob.glob(args.model + '*.pth')):
			if os.path.isdir(args.content_image):
		# for i in os.walk(args.content_image):
				for i in tqdm(glob.glob(args.content_image + '*.*')):

					if args.tile != None:
						tiles = image_slicer.slice(i, args.tile, save=True)
					# 	# tiles = image_slicer.open_images(args.tmp + '/')
					# 	# image_slicer.save_tiles(tiles,directory=args.tmp)


					print("\n\n" + str(i))
					savdir = str(args.output_image[:] + "/" + os.path.basename(modX)[:-3] +  os.path.basename(i))
					print(savdir)
					# sys.stdout.write(savdir)
					sys.stdout.flush()
					if args.tile != None:
						for tile in tiles:
							# print("\n\n TILE FILENAME")
							print(tile.filename)
							# content_image = utils.load_image(i, scale=args.content_scale)
							content_image = tile.image
							# grad = tiled_gradient(gradient=,image=args.content_image)
							if args.max_size != None:
								content_image = resize(content_image, args.max_size)
							content_transform = transforms.Compose([
								transforms.ToTensor(),
								transforms.Lambda(lambda x: x.mul(255))
							])
							content_image = content_transform(content_image)
							content_image = content_image.unsqueeze(0)
							if args.cuda:
								content_image = content_image.cuda()
							content_image = Variable(content_image, volatile=True)

							style_model = TransformerNet()
							style_model.load_state_dict(torch.load(modX))
							if args.cuda:
								style_model.cuda()

							output0 = style_model(content_image)
							output = style_model(output0)
							if args.cuda:
								output = output.cpu()
							output_data = output.data[0]

							tile.image = output_data

							utils.save_image(tile.filename, tile.image)
							# image_slicer.save_tiles(tiles)
					
					if args.tile != None:	
						tiles = image_slicer.open_images(args.tmp)
						image_slicer.save_tiles(tiles)
						output = image_slicer.join(tiles)
						output.save(savdir)





	# 				content_image = utils.load_image(i, scale=args.content_scale)
	# 				# grad = tiled_gradient(gradient=,image=args.content_image)
	# 				if args.max_size != None:
	# 					content_image = resize(content_image, args.max_size)
	# 				content_transform = transforms.Compose([
	# 					transforms.ToTensor(),
	# 					transforms.Lambda(lambda x: x.mul(255))
	# 				])
	# 				content_image = content_transform(content_image)
	# 				content_image = content_image.unsqueeze(0)
	# 				if args.cuda:
	# 					content_image = content_image.cuda()
	# 				content_image = Variable(content_image, volatile=True)

	# 				style_model = TransformerNet()
	# 				style_model.load_state_dict(torch.load(modX))
	# 				if args.cuda:
	# 					style_model.cuda()

	# 				output0 = style_model(content_image)
	# 				output = style_model(output0)
	# 				if args.cuda:
	# 					output = output.cpu()
	# 				output_data = output.data[0]

	# 				utils.save_image(savdir, output_data)
	# 		else:
	# 			if args.max_size is not None:
	# 				content_image = resize(args.content_image, args.max_size)
	# 			content_image = utils.load_image(args.content_image, scale=args.content_scale)
	# 			content_transform = transforms.Compose([
	# 				transforms.ToTensor(),
	# 				transforms.Lambda(lambda x: x.mul(255))
	# 			])
	# 			content_image = content_transform(content_image)
	# 			content_image = content_image.unsqueeze(0)
	# 			if args.cuda:
	# 				content_image = content_image.cuda()
	# 			content_image = Variable(content_image, volatile=True)

	# 			style_model = TransformerNet()
	# 			style_model.load_state_dict(torch.load(modX))
	# 			if args.cuda:
	# 				style_model.cuda()

	# 			output0 = style_model(content_image)
	# 			output = style_model(output0)
	# 			if args.cuda:
	# 				output = output.cpu()
	# 			output_data = output.data[0]

	# 			utils.save_image(args.output_image, output_data)
	# 			return

	# if os.path.isdir(args.content_image):
	# 	# for i in os.walk(args.content_image):
	# 	for i in tqdm(glob.glob(args.content_image + '*.*')):
	# 		print("\n\n" + str(i))
	# 		savdir = str(args.output_image[:] + "/" + os.path.basename(args.model)[:-3] +  os.path.basename(i))
	# 		print(savdir)
	# 		sys.stdout.write(savdir)
	# 		sys.stdout.flush()
	# 		content_image = utils.load_image(i, scale=args.content_scale)
	# 		if args.max_size is not None:
	# 			content_image = resize(content_image, args.max_size)
	# 		content_transform = transforms.Compose([
	# 			transforms.ToTensor(),
	# 			transforms.Lambda(lambda x: x.mul(255))
	# 		])
	# 		content_image = content_transform(content_image)
	# 		content_image = content_image.unsqueeze(0)
	# 		if args.cuda:
	# 			content_image = content_image.cuda()
	# 		content_image = Variable(content_image, volatile=True)

	# 		style_model = TransformerNet()
	# 		style_model.load_state_dict(torch.load(args.model))
	# 		if args.cuda:
	# 			style_model.cuda()

	# 		output0 = style_model(content_image)
	# 		output = style_model(output0)
	# 		if args.cuda:
	# 			output = output.cpu()
	# 		output_data = output.data[0]

	# 		utils.save_image(savdir, output_data)
	# else:
		
	# 	content_image = utils.load_image(args.content_image, scale=args.content_scale)
	# 	if args.max_size is not None:
	# 		content_image = resize(content_image, args.max_size)
	# 	content_transform = transforms.Compose([
	# 		transforms.ToTensor(),
	# 		transforms.Lambda(lambda x: x.mul(255))
	# 	])
	# 	content_image = content_transform(content_image)
	# 	content_image = content_image.unsqueeze(0)
	# 	if args.cuda:
	# 		content_image = content_image.cuda()
	# 	content_image = Variable(content_image, volatile=True)

	# 	style_model = TransformerNet()
	# 	style_model.load_state_dict(torch.load(args.model))
	# 	if args.cuda:
	# 		style_model.cuda()

	# 	output0 = style_model(content_image)
	# 	output = style_model(output0)
	# 	if args.cuda:
	# 		output = output.cpu()
	# 	output_data = output.data[0]

	# 	utils.save_image(args.output_image, output_data)


def main():
	main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
	subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

	video_arg_parser = subparsers.add_parser("video", help="parser for video arguments")
	video_arg_parser.add_argument("--content-video", type=str, required=True,
								 help="path to content video you want to stylize")

	video_arg_parser.add_argument("--mode",type=int,default=0,
								 help="0 = use image sequence folders\n1 = use a file with ffmpeg\n2 = use a file with VideoSequence")

	video_arg_parser.add_argument("--content-scale", type=int, default=None,
								 help="factor for scaling down the content image")
	video_arg_parser.add_argument("--max-size", type=int, default=None,
								 help="maximum pixel resolution")
	video_arg_parser.add_argument("--fps", type=str, required=True,
								 help="Framerate or Frames Per Second ")

	video_arg_parser.add_argument("--output-video", type=str, required=True,
								 help="path for saving the output video")
	
	video_arg_parser.add_argument("--tmp", type=str, required=True,
								 help="tmp directory for processing")

	video_arg_parser.add_argument("--model", type=str, required=True,
								 help="saved model to be used for stylizing the image")
	
	video_arg_parser.add_argument("--cuda", type=int, required=True,
								 help="set it to 1 for running on GPU, 0 for CPU")

####################################################################

	train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
	train_arg_parser.add_argument("--epochs", type=int, default=2,
								  help="number of training epochs, default is 2")
	train_arg_parser.add_argument("--batch-size", type=int, default=1,
								  help="batch size for training, default is 1")
	train_arg_parser.add_argument("--dataset", type=str, required=True,
								  help="path to training dataset, the path should point to a folder "
									   "containing another folder with all the training images")
	train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
								  help="path to style-image")
	train_arg_parser.add_argument("--save-model-path", type=str, required=True,
								  help="path to file where trained model will be saved.")
	train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
								  help="path to folder where checkpoints of trained models will be saved")
	train_arg_parser.add_argument("--image-size", type=int, default=256,
								  help="size of training images, default is 256 X 256")
	train_arg_parser.add_argument("--style-size", type=int, default=None,
								  help="size of style-image, default is the original size of style image")
	train_arg_parser.add_argument("--cuda", type=int, required=True,
								  help="set it to 1 for running on GPU, 0 for CPU")
	train_arg_parser.add_argument("--seed", type=int, default=42,
								  help="random seed for training")
	train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
								  help="weight for content-loss, default is 1e5")
	train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
								  help="weight for style-loss, default is 1e10")
	train_arg_parser.add_argument("--lr", type=float, default=1e-3,
								  help="learning rate, default is 1e-3")
	train_arg_parser.add_argument("--log-interval", type=int, default=5000,
								  help="number of images after which the training loss is logged, default is 500")
	train_arg_parser.add_argument("--checkpoint-interval", type=int, default=20000,
								  help="number of batches after which a checkpoint of the trained model will be created")
	train_arg_parser.add_argument("--init-from", type=str)
####################################################################

	eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
	eval_arg_parser.add_argument("--content-image", type=str, required=True,
								 help="path to content image you want to stylize")
	eval_arg_parser.add_argument("--tile", type=int, default=None,
								 help="number of tiles for tiled processing")
	eval_arg_parser.add_argument("--tmp", type=str, default=None,
								 help="tmp directory, required for --tile")
	eval_arg_parser.add_argument("--content-scale", type=int, default=None,
								 help="factor for scaling down the content image")
	eval_arg_parser.add_argument("--max-size", type=int, default=None,
								 help="maximum pixel resolution")
	eval_arg_parser.add_argument("--output-image", type=str, required=True,
								 help="path for saving the output image")
	
	eval_arg_parser.add_argument("--model", type=str, required=True,
								 help="saved model to be used for stylizing the image")
	
	eval_arg_parser.add_argument("--cuda", type=int, required=True,
								 help="set it to 1 for running on GPU, 0 for CPU")

	args = main_arg_parser.parse_args()

	if args.subcommand is None:
		print("ERROR: specify either video ortrain or eval")
		sys.exit(1)
	if args.cuda and not torch.cuda.is_available():
		print("ERROR: cuda is not available, try running on CPU")
		sys.exit(1)

	if args.subcommand == "train":
		check_paths(args)
		train(args)
	elif args.subcommand == "eval":
		stylize(args)
	elif args.subcommand == "video":
		# check_paths(args)
		transform_video(args)

if __name__ == "__main__":
	main()
	print("ALL DONE!")