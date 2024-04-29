import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import os
import json
import collections
import torch
import torchvision
import torchvision.models as models
from torchvision import transforms
from PIL import Image as pilImage

COLOR_JITTER_MODEL = {
    'MODEL': 'config/model/colorjittermodel.pt',
    'PRECOMPUTED': 'config/colorjitterprecomputed.json',
}
FLIPPING_MODEL = {
    'MODEL': 'config/model/editedresnetmodel.pt',
    'PRECOMPUTED': 'config/editedresprecomputed.json',
}
PRECOMPUTED_IMAGES = 'config/precomputed_rank.json'
IMAGES_DIRECTORY = 'dataset/VOCdevkit/VOC2012/JPEGImages/'

CLASS_MAPPING = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class VOCApplication(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.input_img_name = 'No image file selected yet'
        self.model_selected = 'Color Jitter Model'
        self.model_in_use = models.resnet18(pretrained=False)
        num_ftrs = self.model_in_use.fc.in_features
        self.model_in_use.fc = torch.nn.Linear(num_ftrs, 20)
        self.model_in_use.load_state_dict(torch.load(COLOR_JITTER_MODEL['MODEL'], map_location='cpu'), strict=False)
        self.model_mapping = {
            'Color Jitter Model': COLOR_JITTER_MODEL, 'Flipping Model': FLIPPING_MODEL}
        with open(PRECOMPUTED_IMAGES, 'r') as pc:
            self.precomputed = json.load(pc)
        self.class_mapping = CLASS_MAPPING
        self.values = {}
        with open(FLIPPING_MODEL['PRECOMPUTED'], 'r') as pcf:
            self.values['Edited Resnet Model']= json.load(pcf)
        with open(COLOR_JITTER_MODEL['PRECOMPUTED'], 'r') as pcc:
            self.values['Color Jitter Model'] = json.load(pcc)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.frame1 = tk.Frame(self)
        self.img_name_display = tk.Label(self.frame1)
        self.img_name_display['relief'] = 'sunken'
        self.img_name_display['text'] = self.input_img_name
        self.img_name_display['wraplength'] = 500
        self.img_name_display.pack(side='left')
        self.select_img = tk.Button(self.frame1)
        self.select_img['text'] = 'Choose Image'
        self.select_img['command'] = self.select_img_cmd
        self.select_img.pack(side='right')
        self.frame1.pack(side='top')

        self.frame2 = tk.Frame(self)
        self.var = tk.IntVar()
        self.model_choice1 = tk.Radiobutton(self.frame2)
        self.model_choice1['text'] = 'Color Jitter Model'
        self.model_choice1['command'] = self.select_model
        self.model_choice1['variable'] = self.var
        self.model_choice1['value'] = 1
        self.model_choice1.pack(side='left')
        self.model_choice2 = tk.Radiobutton(self.frame2)
        self.model_choice2['text'] = 'Edited Resnet Model'
        self.model_choice2['command'] = self.select_model
        self.model_choice2['variable'] = self.var
        self.model_choice2['value'] = 2
        self.model_choice2.pack(side='right')
        self.var.set(1)
        self.frame2.pack(side='top')

        self.frame3 = tk.Frame(self)
        self.just_words = tk.Label(self.frame3)
        self.just_words['text'] = 'Precomputed, Class'
        self.just_words.pack(side='left')
        self.class_choice = tk.StringVar()
        self.class_dict = ['None'] + [str(i) for i in range(1,21)]
        self.img_choice = tk.StringVar()
        self.img_dict = self.precomputed['Color Jitter Model']['Possible']
        #####
        self.fake_dict = ['new1', 'new2']
        #####
        self.class_roller = tk.OptionMenu(self.frame3, self.class_choice, *self.class_dict, command=self.orderoutputs)
        self.class_choice.set(self.class_dict[0])
        self.img_roller = tk.OptionMenu(self.frame3, self.img_choice, *self.img_dict)
        self.img_choice.set(self.img_dict[0])
        self.class_roller.pack(side='left')
        self.precomputed_button = tk.Button(self.frame3)
        self.precomputed_button['text'] = "View"
        self.precomputed_button['command'] = self.show_precomputed
        self.precomputed_button.pack(side='right')
        self.threshold = tk.Entry(self.frame3)
        self.threshold.delete(0, 'end')
        self.threshold.insert(0, "0.7")
        self.threshold.pack(side='right')
        self.img_roller.pack(side='right')
        self.frame3.pack(side='top')

        self.img_display = tk.Label(self)
        self.img_display.pack(side='top')

        self.target_display = tk.Label(self)
        self.target_display.pack(side='top')
        self.pred_display = tk.Label(self)
        self.pred_display.pack(side='top')

        self.analyse = tk.Button(self)
        self.analyse['text'] = 'Start Analysis'
        self.analyse['command'] = self.analyse_img
        self.analyse.pack(side='top')

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def analyse_img(self):
        try:
            threshold = float(self.threshold.get())
            if threshold > 1 or threshold < 0:
                messagebox.showerror("Error", "Threshold must be a value from 0 to 1")
            else:
                self.pred_display['text'] = 'Pred: Analysis in Progress'
                img = pilImage.open(self.input_img_name)
                fivecrop_testtime = transforms.Compose([
                    transforms.Resize(280),
                    transforms.FiveCrop(224), 
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Lambda(lambda norms: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])(norm) for norm in norms]))
                ])
                self.model_in_use.eval()
                with torch.no_grad():
                    img_tensor = fivecrop_testtime(img)
                    output = self.model_in_use(img_tensor)
                    output = output.mean(0)
                    output = torch.sigmoid(output)
                    pred_label = ''
                    for i in range(len(output)):
                        if output[i] > threshold:
                            pred_label += self.class_mapping[i] + ' '
                    self.target_display['text'] = ''
                    self.pred_display['text'] = 'Pred: ' + pred_label
        except Exception as e:
            print(e)
            messagebox.showerror("Error", "Threshold must be a value from 0 to 1")
        

    def select_img_cmd(self):
        self.input_img_name = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(
            ('png files', '*.png'), ("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.img_name_display['text'] = self.input_img_name
        self.img = Image.open(self.input_img_name)
        self.img = ImageTk.PhotoImage(self.img)
        self.img_display['image'] = self.img
        self.target_display['text'] = ''
        self.pred_display['text'] = 'Pred: Click Start Analysis'
        return

    def select_model(self):
        selection = self.var.get()
        if selection == 1:
            self.model_selected = 'Color Jitter Model'
            self.model_in_use = models.resnet18(pretrained=False)
            num_ftrs = self.model_in_use.fc.in_features
            self.model_in_use.fc = torch.nn.Linear(num_ftrs, 20)
            self.model_in_use.load_state_dict(torch.load(COLOR_JITTER_MODEL['MODEL'], map_location='cpu'), strict=False)
        else:
            self.model_selected = 'Edited Resnet Model'
            self.model_in_use = models.resnet18(pretrained=False)
            num_ftrs = self.model_in_use.fc.in_features
            self.model_in_use.fc = torch.nn.Linear(num_ftrs, 20)
            self.model_in_use.load_state_dict(torch.load(FLIPPING_MODEL['MODEL'], map_location='cpu'), strict=False)
        return

    def orderoutputs(self, choice):
        if self.class_choice.get() == 'None':
            self.img_roller['menu'].delete(0, 'end')
            
            for choice in self.precomputed[self.model_selected]['Possible']:
                self.img_roller['menu'].add_command(label=choice, command=tk._setit(self.img_choice, choice))
            self.img_choice.set(self.img_dict[0])
        else:
            self.img_roller['menu'].delete(0, 'end')
            
            for choice in self.precomputed[self.model_selected][self.class_choice.get()]:
                self.img_roller['menu'].add_command(label=choice, command=tk._setit(self.img_choice, choice))
            self.img_choice.set(self.precomputed[self.model_selected][self.class_choice.get()][0])

    def show_precomputed(self):
        try:
            threshold = float(self.threshold.get())
            if threshold > 1 or threshold < 0:
                messagebox.showerror("Error", "Threshold must be a value from 0 to 1")
            else:
                path = IMAGES_DIRECTORY + self.img_choice.get() + '.jpg'
                self.img = Image.open(path)
                self.img = ImageTk.PhotoImage(self.img)
                self.img_display['image'] = self.img
                pred_values = self.values[self.model_selected][self.img_choice.get()]['Pred']
                true_values = self.values[self.model_selected][self.img_choice.get()]['True']
                pred_label = ''
                true_label = ''
                for i in range(len(pred_values)):
                    if pred_values[i] > threshold:
                        pred_label += self.class_mapping[i] + ' '
                for j in range(len(true_values)):
                    if true_values[j] == 1:
                        true_label += self.class_mapping[j] + ' '
                self.target_display['text'] = 'True: ' + true_label
                self.pred_display['text'] = 'Pred: ' + pred_label
        except Exception as e:
            print(e)
            messagebox.showerror("Error", "Threshold must be a value from 0 to 1")
        


root = tk.Tk()
root.title('VOC Demo')
root.geometry('1080x720')
app = VOCApplication(master=root)
app.mainloop()
