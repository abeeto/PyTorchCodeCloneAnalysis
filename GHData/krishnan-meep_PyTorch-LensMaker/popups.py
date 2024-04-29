from tkinter import *
from tkinter import ttk
import re

class LinearPopup(object):
	def __init__(self,master):
		top =self.top = Toplevel(master)
		self.l1 = Label(top, text = "Number of input nodes: ")
		self.l2 = Label(top, text = "Number of output nodes: ")
		self.in_entry = Entry(top, width = 5)
		self.out_entry = Entry(top, width = 5)
		self.apply = False

		self.bias_var = IntVar()
		self.spec_var = IntVar()

		Checkbutton(top, text = "Apply SpectralNorm", variable = self.spec_var).grid(row = 2, column = 0)
		Checkbutton(top, text = "Use Bias", variable = self.bias_var).grid(row = 2, column = 1)

		self.b = Button(top,text='Add Layer',command=self.cleanup)

		self.l1.grid(row = 0, column = 0)
		self.l2.grid(row = 1, column = 0)
		self.in_entry.grid(row = 0, column = 1)
		self.out_entry.grid(row = 1, column = 1)
		self.b.grid(row = 2, column = 2)

	def cleanup(self):
		self.in_nodes = self.in_entry.get()
		self.out_nodes = self.out_entry.get()

		if not self.in_nodes or not self.out_nodes:
			self.top.destroy()
			return

		self.spec_set = self.spec_var.get()
		self.bias_set = self.bias_var.get()

		self.apply = True
		self.top.destroy()

class ConvolutionPopup(object):
	def __init__(self,master):
		top =self.top = Toplevel(master)
		self.l1 = Label(top, text = "Number of input filters: ")
		self.l2 = Label(top, text = "Number of output filters: ")
		self.l3 = Label(top, text = "Kernel Size: ")
		self.l4 = Label(top, text = "Stride: ")
		self.l5 = Label(top, text = "Padding: ")
		self.in_entry = Entry(top, width = 5)
		self.out_entry = Entry(top, width = 5)
		self.kernel_entry = Entry(top, width = 5)
		self.stride_entry = Entry(top, width = 5)
		self.padding_entry = Entry(top, width = 5)
		self.drop1 = ttk.Combobox(top, values = ["1d", "2d", "3d"], state = "readonly", width = 6)
		self.drop1.current(0)
		self.apply = False

		self.bias_var = IntVar(value = 1)
		self.spec_var = IntVar()

		Checkbutton(top, text = "Apply SpectralNorm", variable = self.spec_var).grid(row = 5, column = 0)
		Checkbutton(top, text = "Use Bias", variable = self.bias_var).grid(row = 5, column = 1)

		self.b = Button(top,text='Add Layer',command=self.cleanup)

		self.l1.grid(row = 0, column = 0)
		self.l2.grid(row = 1, column = 0)
		self.l3.grid(row = 2, column = 0)
		self.l4.grid(row = 3, column = 0)
		self.l5.grid(row = 4, column = 0)
		self.in_entry.grid(row = 0, column = 1)
		self.drop1.grid(row = 0, column = 2)
		self.out_entry.grid(row = 1, column = 1)
		self.kernel_entry.grid(row = 2, column = 1)
		self.stride_entry.grid(row = 3, column = 1)
		self.padding_entry.grid(row = 4, column = 1)
		self.b.grid(row = 5, column = 2)

	def cleanup(self):
		self.in_filters = self.in_entry.get()
		self.out_filters = self.out_entry.get()
		self.dim = self.drop1.get()
		self.kernel_size = self.kernel_entry.get()
		self.stride = self.stride_entry.get()
		self.padding = self.padding_entry.get()

		if not self.kernel_size: self.kernel_size = 3 
		if not self.stride: self.stride = 1 
		if not self.padding: self.padding = 1 

		if not self.in_filters or not self.out_filters:
			self.top.destroy()
			return

		self.spec_set = self.spec_var.get()
		self.bias_set = self.bias_var.get()

		self.apply = True
		self.top.destroy()

class ActivationPopup(object):
	def __init__(self, master, activations_list):
		top =self.top = Toplevel(master)
		self.drop1 = ttk.Combobox(top, values = activations_list, state = "readonly", width = 20)
		self.button = Button(top, text = "Add", command = self.cleanup)
		self.apply = False

		self.drop1.bind("<<ComboboxSelected>>", self.change_activation)
		self.drop1.current(0)
		self.curr_activation = activations_list[0]

		self.drop1.grid(row = 0, column = 0, padx = 5, pady = 5)
		self.button.grid(row = 0, column = 1, padx = 5, pady = 5)

	def change_activation(self, event = None):
		self.curr_activation = event.widget.get()

	def cleanup(self):
		self.apply = True
		self.top.destroy()

class UpsamplePopup(object):
	def __init__(self, master, upsample_list):
		top =self.top = Toplevel(master)
		self.drop1 = ttk.Combobox(top, values = upsample_list, state = "readonly", width = 20)
		self.button = Button(top, text = "Add", command = self.cleanup)
		self.apply = False

		self.drop1.bind("<<ComboboxSelected>>", self.change_upsample)
		self.drop1.current(0)
		self.curr_upsample = upsample_list[0]

		self.drop1.grid(row = 0, column = 0, padx = 5, pady = 5)
		self.button.grid(row = 0, column = 1, padx = 5, pady = 5)

	def change_upsample(self, event = None):
		self.curr_upsample = event.widget.get()

	def cleanup(self):
		self.apply = True
		self.top.destroy()

class ReshapePopup(object):
	def __init__(self, master):
		top = self.top = Toplevel(master)
		self.l1 = Label(top, text = "Specify output dimensions as D1, D2, D3...")
		self.entry = Entry(top, width = 20)
		self.button = Button(top, text = "Add", command = self.cleanup)

		self.apply = False

		self.l1.grid(row = 0, column = 0, padx = 5, pady = 5)
		self.entry.grid(row = 1, column = 0, padx = 5, pady = 5)
		self.button.grid(row = 1, column = 1, padx = 5, pady = 5)

		self.matcher = re.compile("(\d*,)*\d+")
		self.result = ""

	def cleanup(self):
		if self.matcher.match(self.entry.get()) is None:
			self.top.destroy()
			return

		self.result = self.entry.get()
		self.apply = True
		self.top.destroy()

class PoolPopup(object):
	def __init__(self, master):
		top = self.top = Toplevel(master)
		self.drop1 = ttk.Combobox(top, values = ["MaxPool", "AvgPool"], state = "readonly", width = 16)
		self.drop1.current(0)
		self.drop2 = ttk.Combobox(top, values = ["1d", "2d", "3d"], state = "readonly", width = 6)
		self.drop2.current(0)
		self.button = Button(top, text = "Add", command = self.cleanup)

		self.l1 = Label(top, text = "Kernel Size: ")
		self.l2 = Label(top, text = "Stride: ")
		self.l3 = Label(top, text = "Padding: ")
		self.kernel_entry = Entry(top, width = 5)
		self.stride_entry = Entry(top, width = 5)
		self.padding_entry = Entry(top, width = 5)
		self.apply = False

		self.l1.grid(row = 1, column = 0)
		self.l2.grid(row = 2, column = 0)
		self.l3.grid(row = 3, column = 0)
		self.drop1.grid(row = 0, column = 0)
		self.drop2.grid(row = 0, column = 1)
		self.kernel_entry.grid(row = 1, column = 1)
		self.stride_entry.grid(row = 2, column = 1)
		self.padding_entry.grid(row = 3, column = 1)
		self.button.grid(row = 4, column = 2)

	def cleanup(self):
		self.pool_type = self.drop1.get()
		self.dim = self.drop2.get()
		self.kernel_size = self.kernel_entry.get()
		self.stride = self.stride_entry.get()
		self.padding = self.padding_entry.get()

		if not self.kernel_size: self.kernel_size = 3 
		if not self.stride: self.stride = 2 
		if not self.padding: self.padding = 1 

		self.apply = True
		self.top.destroy()

class SkipPopup(object):
	def __init__(self, master, num_list):
		top =self.top = Toplevel(master)
		self.drop1 = ttk.Combobox(top, values = num_list, state = "readonly", width = 20)
		self.button = Button(top, text = "Add Skip from this slot", command = self.cleanup)
		self.apply = False

		self.drop1.bind("<<ComboboxSelected>>", self.change_upsample)
		self.drop1.current(0)
		self.curr_slot = num_list[0]

		self.drop1.grid(row = 0, column = 0, padx = 5, pady = 5)
		self.button.grid(row = 0, column = 1, padx = 5, pady = 5)

	def change_upsample(self, event = None):
		self.curr_slot = event.widget.get()

	def cleanup(self):
		self.apply = True
		self.top.destroy()