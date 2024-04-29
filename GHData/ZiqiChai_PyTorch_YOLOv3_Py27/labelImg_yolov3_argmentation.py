# -*- coding: UTF-8 -*-
import sys
import os
import shutil
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
	sys.path.remove(ros_path)
import cv2 as cv
import numpy as np
import random


def win2unix(input_filename, output_filename):
    assert input_filename and os.path.exists(input_filename), "input file not found"
    with open(input_filename, "rb") as f:
        data = bytearray(os.path.getsize(input_filename))
        f.readinto(data)
        print(data)
        data = data.replace(b"\r\n", b"\n")
        print(data)
    with open(output_filename, "wb") as f:
        f.write(data)


def gen_origin_txt_label(infile, outfile):
	with open(infile, "r") as ifs:
		with open(outfile,"w") as ofs:
			for line in ifs.readlines():
				line=line.strip('\n')
				line=line.split(" ")
				if len(line)==5:
					print("wite:{} to {}".format(line,outfile))
					ofs.write(line[0])
					ofs.write(" ")
					ofs.write(line[1])
					ofs.write(" ")
					ofs.write(line[2])
					ofs.write(" ")
					ofs.write(line[3])
					ofs.write(" ")
					ofs.write(line[4])
					ofs.write("\n")


def gen_flip_x_txt_label(infile, outfile):
	with open(infile, "r") as ifs:
		with open(outfile,"w") as ofs:
			for line in ifs.readlines():
				line=line.strip('\n')
				line=line.split(" ")
				if len(line)==5:
					print("wite:{} to {}".format(line,outfile))
					ofs.write(line[0])
					ofs.write(" ")
					ofs.write(line[1])
					ofs.write(" ")
					ofs.write(str(round(1-float(line[2]), 6)))
					ofs.write(" ")
					ofs.write(line[3])
					ofs.write(" ")
					ofs.write(line[4])
					ofs.write("\n")


def gen_flip_y_txt_label(infile, outfile):
	with open(infile, "r") as ifs:
		with open(outfile,"w") as ofs:
			for line in ifs.readlines():
				line=line.strip('\n')
				line=line.split(" ")
				if len(line)==5:
					print("wite:{} to {}".format(line,outfile))
					ofs.write(line[0])
					ofs.write(" ")
					ofs.write(str(round(1-float(line[1]), 6)))
					ofs.write(" ")
					ofs.write(line[2])
					ofs.write(" ")
					ofs.write(line[3])
					ofs.write(" ")
					ofs.write(line[4])
					ofs.write("\n")


def gen_flip_x_y_txt_label(infile, outfile):
	with open(infile, "r") as ifs:
		with open(outfile,"w") as ofs:
			for line in ifs.readlines():
				line=line.strip('\n')
				line=line.split(" ")
				if len(line)==5:
					print("wite:{} to {}".format(line,outfile))
					ofs.write(line[0])
					ofs.write(" ")
					ofs.write(str(round(1-float(line[1]), 6)))
					ofs.write(" ")
					ofs.write(str(round(1-float(line[2]), 6)))
					ofs.write(" ")
					ofs.write(line[3])
					ofs.write(" ")
					ofs.write(line[4])
					ofs.write("\n")


class DataAugment(object):
	def __init__(self, img_read_name="", img_format='png', img_save_prefix="./image", img_save_index=0):
		self.img_read_name_no_postfix = img_read_name
		self.img_save_name  = img_save_prefix               # Prefix name is image
		self.add_saltNoise  = True                          # False
		self.gaussianBlur   = True                          # False
		self.changeExposure = True                          # False
		self.id = img_save_index
		self.format = '.' + img_format
		img = cv.imread(self.img_read_name_no_postfix+self.format)
		try:
			img.shape
		except:
			print('No Such image!---'+self.img_read_name_no_postfix+self.format)
			sys.exit(0)
		self.src = img
		dst1 = cv.flip(img, 0, dst=None)
		dst2 = cv.flip(img, 1, dst=None)
		dst3 = cv.flip(img, -1, dst=None)
		self.flip_x = dst1
		self.flip_y = dst2
		self.flip_x_y = dst3
		cv.imwrite(self.img_save_name+str(self.id)+self.format, self.src)
		cv.imwrite(self.img_save_name+str(self.id)+'_flip_x'+self.format, self.flip_x)
		cv.imwrite(self.img_save_name+str(self.id)+'_flip_y'+self.format, self.flip_y)
		cv.imwrite(self.img_save_name+str(self.id)+'_flip_x_y'+self.format, self.flip_x_y)

	def gaussian_blur_fun(self):
		if self.gaussianBlur:
			dst1 = cv.GaussianBlur(self.src, (5, 5), 0)
			dst2 = cv.GaussianBlur(self.flip_x, (5, 5), 0)
			dst3 = cv.GaussianBlur(self.flip_y, (5, 5), 0)
			dst4 = cv.GaussianBlur(self.flip_x_y, (5, 5), 0)
			cv.imwrite(self.img_save_name+str(self.id)+'_Gaussian'+self.format, dst1)
			cv.imwrite(self.img_save_name+str(self.id)+'_flip_x'+'_Gaussian'+self.format, dst2)
			cv.imwrite(self.img_save_name+str(self.id)+'_flip_y'+'_Gaussian'+self.format, dst3)
			cv.imwrite(self.img_save_name+str(self.id)+'_flip_x_y'+'_Gaussian'+self.format, dst4)

	def change_exposure_fun(self):
		if self.changeExposure:
			reduce = 0.4
			increase = 1.2
			g = 10
			h, w, ch = self.src.shape
			add  = np.zeros([h, w, ch], self.src.dtype)
			dst1 = cv.addWeighted(self.src, reduce, add, 1-reduce, g)
			dst2 = cv.addWeighted(self.src, increase, add, 1-increase, g)
			dst3 = cv.addWeighted(self.flip_x, reduce, add, 1 - reduce, g)
			dst4 = cv.addWeighted(self.flip_x, increase, add, 1 - increase, g)
			dst5 = cv.addWeighted(self.flip_y, reduce, add, 1 - reduce, g)
			dst6 = cv.addWeighted(self.flip_y, increase, add, 1 - increase, g)
			dst7 = cv.addWeighted(self.flip_x_y, reduce, add, 1 - reduce, g)
			dst8 = cv.addWeighted(self.flip_x_y, increase, add, 1 - increase, g)
			cv.imwrite(self.img_save_name+str(self.id)+'_ReduceEp'+self.format, dst1)
			cv.imwrite(self.img_save_name+str(self.id)+'_flip_x'+'_ReduceEp'+self.format, dst3)
			cv.imwrite(self.img_save_name+str(self.id)+'_flip_y'+'_ReduceEp'+self.format, dst5)
			cv.imwrite(self.img_save_name+str(self.id)+'_flip_x_y'+'_ReduceEp'+self.format, dst7)
			cv.imwrite(self.img_save_name+str(self.id)+'_IncreaseEp'+self.format, dst2)
			cv.imwrite(self.img_save_name+str(self.id)+'_flip_x'+'_IncreaseEp'+self.format, dst4)
			cv.imwrite(self.img_save_name+str(self.id)+'_flip_y'+'_IncreaseEp'+self.format, dst6)
			cv.imwrite(self.img_save_name+str(self.id)+'_flip_x_y'+'_IncreaseEp'+self.format, dst8)

	def add_salt_noise(self):
		if self.add_saltNoise:
			percentage = 0.005
			dst1 = self.src
			dst2 = self.flip_x
			dst3 = self.flip_y
			dst4 = self.flip_x_y
			num = int(percentage * self.src.shape[0] * self.src.shape[1])
			for i in range(num):
				rand_x = random.randint(0, self.src.shape[0] - 1)
				rand_y = random.randint(0, self.src.shape[1] - 1)
				if random.randint(0, 1) == 0:
					dst1[rand_x, rand_y] = 0
					dst2[rand_x, rand_y] = 0
					dst3[rand_x, rand_y] = 0
					dst4[rand_x, rand_y] = 0
				else:
					dst1[rand_x, rand_y] = 255
					dst2[rand_x, rand_y] = 255
					dst3[rand_x, rand_y] = 255
					dst4[rand_x, rand_y] = 255
			cv.imwrite(self.img_save_name+str(self.id)+'_Salt'+self.format, dst1)
			cv.imwrite(self.img_save_name+str(self.id)+'_flip_x'+'_Salt'+self.format, dst2)
			cv.imwrite(self.img_save_name+str(self.id)+'_flip_y'+'_Salt'+self.format, dst3)
			cv.imwrite(self.img_save_name+str(self.id)+'_flip_x_y'+'_Salt'+self.format, dst4)

	def txt_label_generation(self):
		# win2unix(self.img_read_name_no_postfix+".txt", self.img_read_name_no_postfix+".txt")
		gen_origin_txt_label(self.img_read_name_no_postfix+".txt",  self.img_save_name+str(self.id)+".txt")
		gen_flip_x_txt_label(self.img_read_name_no_postfix+".txt",  self.img_save_name+str(self.id)+"_flip_x.txt")
		gen_flip_y_txt_label(self.img_read_name_no_postfix+".txt",  self.img_save_name+str(self.id)+"_flip_y.txt")
		gen_flip_x_y_txt_label(self.img_read_name_no_postfix+".txt",self.img_save_name+str(self.id)+"_flip_x_y.txt")

		if self.gaussianBlur:
			gen_origin_txt_label(self.img_read_name_no_postfix+".txt", self.img_save_name+str(self.id)+'_Gaussian'+".txt")
			gen_flip_x_txt_label(self.img_read_name_no_postfix+".txt", self.img_save_name+str(self.id)+'_flip_x'+'_Gaussian'+".txt")
			gen_flip_y_txt_label(self.img_read_name_no_postfix+".txt", self.img_save_name+str(self.id)+'_flip_y' + '_Gaussian'+".txt")
			gen_flip_x_y_txt_label(self.img_read_name_no_postfix+".txt", self.img_save_name+str(self.id)+'_flip_x_y'+'_Gaussian'+".txt")
		if self.changeExposure:
			gen_origin_txt_label(self.img_read_name_no_postfix+".txt", self.img_save_name+str(self.id)+'_ReduceEp'+".txt")
			gen_flip_x_txt_label(self.img_read_name_no_postfix+".txt", self.img_save_name+str(self.id)+'_flip_x'+'_ReduceEp'+".txt")
			gen_flip_y_txt_label(self.img_read_name_no_postfix+".txt", self.img_save_name+str(self.id)+'_flip_y'+'_ReduceEp'+".txt")
			gen_flip_x_y_txt_label(self.img_read_name_no_postfix+".txt", self.img_save_name+str(self.id)+'_flip_x_y'+'_ReduceEp'+".txt")
			gen_origin_txt_label(self.img_read_name_no_postfix+".txt", self.img_save_name+str(self.id)+'_IncreaseEp'+".txt")
			gen_flip_x_txt_label(self.img_read_name_no_postfix+".txt", self.img_save_name+str(self.id)+'_flip_x'+'_IncreaseEp'+".txt")
			gen_flip_y_txt_label(self.img_read_name_no_postfix+".txt", self.img_save_name+str(self.id)+'_flip_y'+'_IncreaseEp'+".txt")
			gen_flip_x_y_txt_label(self.img_read_name_no_postfix+".txt", self.img_save_name+str(self.id)+'_flip_x_y'+'_IncreaseEp'+".txt")
		if self.add_saltNoise:
			gen_origin_txt_label(self.img_read_name_no_postfix+".txt", self.img_save_name+str(self.id)+'_Salt'+".txt")
			gen_flip_x_txt_label(self.img_read_name_no_postfix+".txt", self.img_save_name+str(self.id)+'_flip_x' + '_Salt'+".txt")
			gen_flip_y_txt_label(self.img_read_name_no_postfix+".txt", self.img_save_name+str(self.id)+'_flip_y' + '_Salt'+".txt")
			gen_flip_x_y_txt_label(self.img_read_name_no_postfix+".txt", self.img_save_name+str(self.id)+'_flip_x_y' + '_Salt'+".txt")


def usage_display():
	print('***************************************************')
	print('Usage Display:')
	print('put this .py script under the same folder with original *.png image and yolo *.txt label file')
	print('generate Fliped GaussianBlured ChangeExposure Add_SaltNoise images with label files')
	print('all the images along with their corresponding .txt labels will be renamed indexing from 0')
	print('the output folder is ./output/')
	print('***************************************************')


if __name__ == "__main__":
	usage_display()

	path = "./"
	img_format = 'png'
	output_folder = "./output/"
	if not os.path.exists(output_folder):
	  os.makedirs(output_folder)
	output_name_prefix = output_folder + "image_"
	save_index = 0

	dirs = os.listdir(path)
	for file_path in dirs:
		print(file_path)
		file_name = file_path.split(".")
		print(file_name)
		if file_name[-1] == "txt":
			full_file_path = path + file_name[0] + "." + img_format
			if os.path.exists(full_file_path):
				print(full_file_path)
				img = cv.imread(full_file_path)
				cv.imshow("img",img)
				cv.waitKey(100)
				dataAugmentObject = DataAugment( path + file_name[0], img_format, output_name_prefix, save_index)
				dataAugmentObject.gaussian_blur_fun()
				dataAugmentObject.change_exposure_fun()
				dataAugmentObject.add_salt_noise()
				dataAugmentObject.txt_label_generation()
				save_index = save_index + 1
	cv.destroyAllWindows()