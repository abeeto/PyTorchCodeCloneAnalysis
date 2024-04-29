import os
import re


class Parser(object):
	def __init__(self, class_name, arch_list):
		self.import_lines = ['import torch', 'import torchvision', 'import torch.nn as nn', 'import torch.nn.functional as F', 
							'from torch.nn.utils.spectral_norm import SpectralNorm', '']
		self.class_name = class_name
		self.arch_list = arch_list

		self.module_dict = {"Line" : self.parse_linear, "Conv" : self.parse_conv, "Tran" : self.parse_conv_t, "Upsa" : self.parse_upsample,
							"MaxP" : self.parse_pool, "AvgP" : self.parse_pool, "Skip" : self.parse_skip, "Resh" : self.parse_reshape, 
							"Acti" :self.parse_activation, "Flat" : self.parse_flat}

		self.activations_torch = {"Sigmoid" : "torch.sigmoid", "Tanh" : "torch.tanh", "ReLU" : "F.relu", "Leaky ReLU" : "F.leaky_relu"}


		if not self.class_name:
			self.class_name = "Poggersdude"
		self.class_name = re.sub(r"\s+", '_', self.class_name)

		self.init_lines = ['class ' + self.class_name + '(nn.Module):', 
							" "*4 + "def __init__(self):",
							" "*8 + "super(" + self.class_name + ", self).__init__()"]

		self.forward_lines = ["", 
								" "*4 + "def forward(self, x):",
								" "*8 + "return x"]

		self.linear_count = 0
		self.conv_count = 0
		self.pool_count = 0
		self.conv_t_count = 0
		self.skip_count = 0

	def parse_architecture(self):
		for i in self.arch_list:
			#The Mystical Conjuration ###############################################
			self.module_dict[i[:4]](i)

	def write_to_py(self):
		with open(self.class_name.lower() + ".py", "w") as file:
			for i in [self.import_lines, self.init_lines, self.forward_lines]:
				for j in i:
					print(j)
					file.write(j + "\n")


	def parse_linear(self, text):
		in_dim, out_dim = re.findall(r"\d+", text)
		spec, no_bias = re.findall(r"S", text), re.findall(r"NB", text)

		string = " "*8 + "self.L" + str(self.linear_count) + " = "
		string_f = " "*8 + "x = self.L" + str(self.linear_count) + "(x)"

		if len(spec):
			string += "SpectralNorm("

		string += "nn.Linear(" + in_dim + ", " + out_dim

		if len(no_bias):
			string += ", bias = False"

		string += ")"
		if len(spec):
			string += ")"

		self.linear_count += 1
		self.init_lines.append(string)
		self.forward_lines.insert(-1, string_f)

	def parse_conv(self, text, trans = ""):
		dim, in_filters, out_filters, k, s, p = re.findall(r"\d+", text)
		spec, no_bias = re.findall(r"S", text), re.findall(r"NB", text)

		name = "self.C"
		if trans:
			name = "self.TC"

		string = " "*8 + name + str(self.conv_count) + " = "
		string_f = " "*8 + "x = " + name + str(self.conv_count) + "(x)"

		if len(spec):
			string += "SpectralNorm("

		string += "nn.Conv" + trans + dim + "d(" + in_filters + ", " + out_filters
		string += ", kernel_size = " + k + ", stride = " + s + ", padding = " + p 

		if len(no_bias):
			string += ", bias = False"

		string += ")"
		if len(spec):
			string += ")"

		self.conv_count += 1
		self.init_lines.append(string)
		self.forward_lines.insert(-1, string_f)


	def parse_conv_t(self, text):
		self.parse_conv(text, trans = "Transpose")

	def parse_upsample(self, text):
		text = re.findall(r"\([a-zA-Z\s]+\)", text)[0][1:-1]
		if text[0] == 'N':
			string_f = " "*8 + "x = F.interpolate(x, scale_factor = 2, mode = \"nearest\")"
		elif text[0] == 'B':
			string_f = " "*8 + "x = F.interpolate(x, scale_factor = 2, mode = \"bilinear\")"
		else:
			string = " "*8 + "self.PxShf = nn.PixelShuffle(2)"
			string_f = " "*8 + "x = self.PxShf(x)"

			self.init_lines.append(string)
		self.forward_lines.insert(-1, string_f)

	def parse_pool(self, text):
		dim, k, s, p = re.findall(r"\d+", text)

		string = " "*8 + "self.Pl" + str(self.pool_count) 
		string_pl = " = nn." + text[:9]
		string_pl += "(kernel_size = " + str(k) + ", stride = " + str(s) + ", padding = " + str(p) + ")"

		string += string_pl

		present_check = False
		count = self.pool_count
		for i in self.init_lines:
			if i.endswith(string_pl):
				print("WHAT")
				count = re.findall(r"\d+", i)[0]
				present_check = True

		string_f = " "*8 + "x = self.Pl" + str(count) + "(x)"

		if not present_check:
			self.pool_count += 1
			self.init_lines.append(string)
		self.forward_lines.insert(-1, string_f)

	def parse_skip(self, text):
		pos = re.findall(r"\d+", text)[0]
		string_1 =  " "*8 + "x_s" + str(self.skip_count) + " = x"
		string_2 = " "*8 +"x = x + x_s" + str(self.skip_count)

		self.skip_count += 1
		self.forward_lines.insert(int(pos) + 2, string_1)
		self.forward_lines.insert(-1, string_2)

	def parse_reshape(self, text):
		dim = re.findall(r"\d+", text)
		string = " "*8 + "x = x.view(-1"
		for i in dim:
			string += ", " + i
		string += ")"

		self.forward_lines.insert(-1, string)

	def parse_flat(self, text):
		string = " "*8 + "x = x.view(-1, np.prod(x.shape[1:]))"
		self.forward_lines.insert(-1, string)

	def parse_activation(self, text):
		text = re.findall(r"\([a-zA-Z\s]+\)", text)[0][1:-1]
		string = " "*8 + "x = " + self.activations_torch[text]
		string += "(x)"
		self.forward_lines.insert(-1, string)