from tkinter import *
from tkinter import ttk
from popups import *
from tkinter import messagebox
from arch_parser import Parser  

class LensMaker:
	def __init__(self, root):
		self.root = root

		self.list_modules = ["Linear", "Convolution", "Transposed Convolution", "Upsample", "Pooling", "Skip Connection", "Reshape", "Activation", "Flatten"]
		self.module_popups = [self.add_linear, self.add_conv, self.add_conv_t, self.add_upsample, self.add_pool, self.add_skip, self.add_reshape, self.add_activation, self.add_flatten]
		self.module_func_dict = dict(zip(self.list_modules, self.module_popups))

		self.activations_list = ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU"]
		self.upsample_list = ["Nearest Neighbour", "Bilinear", "Pixel Shuffle"]
		self.curr_layer_index = 0
		#self.internal_arch_list = []
		self.last_linear_nodes = []
		self.last_conv_filters = []
		self.last_dim_check = [0]


		#Things on the left side #############################################################
		self.Frame_L = Frame(root, padx = 5, pady = 5)
		self.name_label = Label(self.Frame_L, text = "Class Name")
		self.name_entry = Entry(self.Frame_L, width = 25)

		self.drop_label = Label(self.Frame_L, text = "Module List")
		self.drop_down = ttk.Combobox(self.Frame_L, values = self.list_modules, state = "readonly", width = 22)
		self.drop_down.current(0)
		self.button_1 = Button(self.Frame_L, text = "Add", command = lambda : self.module_func_dict[self.drop_down.get()]())
		self.button_2 = Button(self.Frame_L, text = "Delete", command = lambda : self.delete_current_layer())

		self.gen_button = Button(self.Frame_L, text = "Generate Code", command = self.generate_code)

		#Things on the right side #############################################################
		self.Frame_R = LabelFrame(root, text = "Architecture", padx = 10, pady = 10)
		self.arch_box = Listbox(self.Frame_R, width = 45, height = 20)
		self.num_box = Listbox(self.Frame_R, width = 3, height = 20)

		self.arch_box.bind("<<ListboxSelect>>", self.change_cursor)
		self.num_box.bind("<<ListboxSelect>>", self.change_cursor)

		self.Frame_L.grid(row = 0, column = 0)
		self.name_label.grid(row = 0, column = 0, pady = 30, padx = 3)
		self.name_entry.grid(row = 0, column = 1, sticky = E)
		self.drop_label.grid(row = 2, column = 0, padx = 3)
		self.drop_down.grid(row = 2, column = 1)
		self.button_1.grid(row = 3, column = 0, pady = 5)
		self.button_2.grid(row = 3, column = 1, padx = 5)
		self.gen_button.grid(row = 4, column = 0, pady = 20)

		self.Frame_R.grid(row = 0, column = 2)
		self.num_box.grid(row = 0, column = 0)
		self.arch_box.grid(row = 0, column = 1, padx = 5)


	def add_linear(self):
		self.w = LinearPopup(self.root)
		self.root.wait_window(self.w.top)
		if not self.w.apply:
			return

		if len(self.last_linear_nodes) and int(self.w.in_nodes) != self.last_linear_nodes[-1]:
			messagebox.showinfo("Oops!","Dimensions from previous layers do not match (" + str(self.last_linear_nodes[-1]) + " != " + self.w.out_nodes + ")")
			return

		self.curr_layer_index +=1
		self.last_linear_nodes.append(int(self.w.out_nodes))

		string = "Linear (" + self.w.in_nodes + ", " + self.w.out_nodes + ") "
		if self.w.spec_set:
			string += "(S)"
		if not self.w.bias_set:
			string += "(NB)"

		self.arch_box.insert(self.curr_layer_index, string)
		self.update_num_box()


	def add_activation(self):
		self.w = ActivationPopup(self.root, self.activations_list)
		self.root.wait_window(self.w.top)
		if not self.w.apply:
			return

		self.curr_layer_index += 1
		self.arch_box.insert(self.curr_layer_index, "Activation (" + self.w.curr_activation + ")")
		self.update_num_box()


	def add_conv(self, transpose = False):
		self.w = ConvolutionPopup(self.root)
		self.root.wait_window(self.w.top)
		if not self.w.apply:
			return

		if len(self.last_conv_filters) and int(self.w.in_filters) != self.last_conv_filters[-1]:
			messagebox.showinfo("Oops!","Dimensions from previous layers do not match (" + str(self.last_conv_filters[-1]) + " != " + self.w.out_filters + ")")
			return

		self.curr_layer_index +=1
		self.last_conv_filters.append(int(self.w.out_filters))

		string = ""
		if transpose:
			string += "Transposed "
		string += "Convolution"

		string += self.w.dim + " (" + self.w.in_filters + ", " + self.w.out_filters
		string += ", k" + str(self.w.kernel_size) + "s" + str(self.w.stride) + "p" + str(self.w.padding) + ")"
		if self.w.spec_set:
			string += "(S)"
		if not self.w.bias_set:
			string += "(NB)"

		self.arch_box.insert(self.curr_layer_index, string)
		self.update_num_box()

	def add_conv_t(self):
		self.add_conv(transpose = True)

	def add_reshape(self):
		self.w = ReshapePopup(self.root)
		self.root.wait_window(self.w.top)
		if not self.w.apply:
			return

		self.curr_layer_index += 1
		self.arch_box.insert(self.curr_layer_index, "Reshape to (" + self.w.result + ")")
		self.update_num_box()

	def add_flatten(self):
		self.curr_layer_index += 1
		self.arch_box.insert(self.curr_layer_index, "Flatten")
		self.update_num_box()

	def add_upsample(self):
		self.w = UpsamplePopup(self.root, self.upsample_list)
		self.root.wait_window(self.w.top)
		if not self.w.apply:
			return

		self.curr_layer_index += 1
		self.arch_box.insert(self.curr_layer_index, "Upsample (" + self.w.curr_upsample + ")")
		self.update_num_box()

	def add_pool(self):
		self.w = PoolPopup(self.root)
		self.root.wait_window(self.w.top)
		if not self.w.apply:
			return

		self.curr_layer_index +=1

		string = self.w.pool_type + self.w.dim
		string += " (k" + str(self.w.kernel_size) + "s" + str(self.w.stride) + "p" + str(self.w.padding) + ")"

		self.arch_box.insert(self.curr_layer_index, string)
		self.update_num_box()

	def add_skip(self):
		self.w = SkipPopup(self.root, ["0"] + list(self.num_box.get(0, END)))
		self.root.wait_window(self.w.top)
		if not self.w.apply:
			return

		self.curr_layer_index += 1
		self.arch_box.insert(self.curr_layer_index, "Skip connection from "+ self.w.curr_slot)
		self.update_num_box()

	def update_num_box(self):
		self.num_box.insert(self.curr_layer_index, str(self.curr_layer_index))
		for i in range(self.curr_layer_index, len(self.num_box.get(0, END))):
			self.num_box.delete(i)
			self.num_box.insert(i, i+1)

	def change_cursor(self, event = None):
		self.curr_layer_index = int(event.widget.curselection()[0])
		print(self.curr_layer_index)

	def delete_current_layer(self):
		item = self.arch_box.get(self.curr_layer_index)
		self.num_box.delete(self.curr_layer_index)
		self.arch_box.delete(self.curr_layer_index)

		########
		#May have to add code to look forward to correct dimensions for Linear and Convolution layers!
		########

		if item[:3] == "Lin":
			self.last_linear_nodes.pop(self.curr_layer_index)
			print(self.last_linear_nodes)

		for i in range(self.curr_layer_index, len(self.num_box.get(0, END))):
			self.num_box.delete(i)
			self.num_box.insert(i, i+1)

		self.curr_layer_index = len(self.num_box.get(0, END))

		#print(self.arch_box.get(0, END))
		#list_box.insert(1, "Spaghetti")

	def generate_code(self):
		self.parser = Parser(self.name_entry.get(), self.arch_box.get(0, END))
		self.parser.parse_architecture()
		self.parser.write_to_py()


root = Tk()
root.geometry("675x420")
#root['bg'] = '#111114'
App = LensMaker(root)
root.mainloop()