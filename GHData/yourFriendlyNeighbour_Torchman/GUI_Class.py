from Tkinter import *
from ttk import *
from PIL import Image, ImageTk
import tkMessageBox
import os
import webbrowser

#Version 05/07/2017



class MainWindow:
    counter = 0

    def __init__(self):
        global photo
        self.root = Tk()
        self.root.title("Pac-Man Game")
        self.list = []
        self.agent = StringVar()
        self.extractor = StringVar()
        self.layout = StringVar()
        self.training = IntVar()
        self.evaluation = IntVar()
        self.quiet = StringVar()

        self.tup_agents = ('KeyboardAgent', 'ApproximateQAgent', 'PacmanQAgent', 'DQLAgent')
        self.tup_layout = ('capsuleClassic', 'contestClassic', 'mediumClassic', 'mediumGrid', 'minimaxClassic', 'openClassic', \
                   'originalClassic', 'smallClassic', 'smallGrid', 'testClassic', 'trappedClassic', 'trickyClassic')
        self.tup_extractor = ('IdentityExtractor', 'CoordinateExtractor', 'BetterExtractor', 'BestExtractor', 'No Extractor')

        name = "/GUI_pic/pacman_keyboard.png"
        path_pic = os.path.dirname(os.path.realpath(__file__))
        photo = ImageTk.PhotoImage(Image.open(path_pic + name).resize((160, 100), Image.ANTIALIAS))
        self.lbl_pic = Label(self.root, image=photo)
        self.lbl_pic.grid(row=4, column=3, columnspan=6, rowspan=4)

    def configurestyles(self):
        bg = 'black'
        fg = 'green'

        self.root.configure(background=bg)
        style = Style()
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TCombobox", background=fg, foreground=fg, fieldbackground=bg, borderthickness=0)
        style.configure("TCheckbutton", background=bg, foreground=fg)
        style.configure("TEntry", background=bg, foreground=fg, borderwidth=1, fieldbackground=bg, borderthickness=0)
        style.configure("TButton", relief='raised', background='#ece74d', foreground=bg)
        style.configure("TMenu", background=bg, foreground=fg)
        return None

    def createElements(self):
        self.createLabels()
        self.createCombo()
        self.createEntrys()
        self.createCheckbuttons()
        return None

    def createLabels(self):
        lbl_agent = Label(self.root, text="Agent:")
        lbl_extract = Label(self.root, text="Extractor:")
        lbl_layout = Label(self.root, text="Choose Layout:")
        lbl_training = Label(self.root, text="Training Rounds(int):")
        lbl_evaluation = Label(self.root, text="Evaluation Rounds(int):")
        lbl_quiet = Label(self.root, text="Quiet Mode:")

        self.list.append(lbl_agent)
        self.list.append(lbl_extract)
        self.list.append(lbl_layout)
        self.list.append(lbl_training)
        self.list.append(lbl_evaluation)
        self.list.append(lbl_quiet)

        return None

    def createCombo(self):
    #Agents
        agents = Combobox(self.root, textvariable=self.agent)
        agents['values'] = self.tup_agents
        agents.set('KeyboardAgent')
        agents.bind("<<ComboboxSelected>>", self.agentchanged)
    #Extractors
        extractors = Combobox(self.root, textvariable=self.extractor)
        extractors['values'] = self.tup_extractor
        self.extractor.set('IdentityExtractor')
    #Layouts
        layouts = Combobox(self.root, textvariable=self.layout)
        layouts['values'] = self.tup_layout
        self.layout.set('mediumClassic')

        self.list[0] = (self.list[0], agents)
        self.list[1] = (self.list[1], extractors)
        self.list[2] = (self.list[2], layouts)

        return None

    def createEntrys(self):
        txt_training = Entry(self.root, textvariable=self.training, width=10)
        self.training.set(0)
       # txt_training.delete(0, END)
        txt_evaluation = Entry(self.root, textvariable=self.evaluation, width=10)
        self.evaluation.set(0)
        #evaluation.delete(0, END)

        self.list[3] = (self.list[3], txt_training)
        self.list[4] = (self.list[4], txt_evaluation)

        return None

    def createCheckbuttons(self):
        cb_quiet = Checkbutton(self.root, variable=self.quiet, onvalue=" -q", offvalue="")
        self.list[5] = (self.list[5], cb_quiet)
        return None

    def createButtons(self, index):
        StartButton = Button(self.root, text="Start Pac-Man", command=self.clickStart)
        StartButton.grid(row=index + 1, column=8, padx=20, pady=20)
        return None

    def createMenu(self):
        menu = Menu(self.root)
        self.root.config(menu=menu)
        menu.add_command(label="Credits", command=self.showCredits)
        menu.config(bg='black', fg='green')


    def orderElements(self):
        i = 0
        padding = 4
        for item in self.list:
            if i in [0, 4]:
                Label(self.root, text='').grid(row=i, column=0, ipady=padding * 2, padx=20)
                i += 1
            item[0].grid(row=i, column=1, sticky=E, pady=padding, padx=1)
            item[1].grid(row=i, column=2, sticky=W, pady=padding)
            i += 1
        self.createButtons(i)
        self.createMenu()
        return None

    def loadimage(self):
        self.setPicture(0)
        return None

    def setPicture(self, index=0):
        name = "/GUI_pic/pacman_neural.png"
        if index == 0:
            name = "/GUI_pic/pacman_keyboard.png"
        path_pic = os.path.dirname(os.path.realpath(__file__))
        self.photo = ImageTk.PhotoImage(Image.open(path_pic + name).resize((160, 100), Image.ANTIALIAS))
        self.lbl_pic.configure(image=self.photo)
        self.lbl_pic.image = self.photo

    def intputsAreValid(self):
        str = ''

        if not self.agent.get() in self.tup_agents: str += 'Selected agent is not valid \n'
        if self.agent.get() != 'KeyboardAgent':
            if not self.extractor.get() in self.tup_extractor: str += 'Selected extractor is not valid \n'
            if type(self.training.get()) == 'int' or self.training.get() < 0: str += 'Training must be a positive Integer \n'
            if type(self.evaluation.get()) == 'int' or self.evaluation.get() < 1: str += 'Evaluation must be a Integer greater 1 \n'
        if not self.layout.get() in self.tup_layout: str += 'Selected layout is not valid \n'
        if len(str) == 0:
            return True
        else:
            tkMessageBox.showinfo("Invalid Parameters", str)
            return  False

#Events
    def agentchanged(self, event):
        if self.agent.get() == 'KeyboardAgent':
            self.setPicture(0)
        else:
            self.setPicture(1)

    def clickStart(self):

        if self.intputsAreValid():
            list = []

            if self.agent.get() != 'KeyboardAgent':
                list.append(" -n " + str(self.training.get() + self.evaluation.get()))
                list.append(self.quiet.get())

                list.append(" -p " + self.agent.get())
                list.append(" -x " + str(self.training.get()))
                if self.extractor != 'NoExtractor': list.append(" -a " + "extractor=" + self.extractor.get())

            list.append(" -l " + self.layout.get())

            options = ''
            for  i in list:
                options += i
            running = "python pacman.py" + options

            print options

            try:
               os.system(running)
            except:
               print 'No'

    def showCredits(self):
        credit = Credits()
        credit.show()


class Credits:

    def __init__(self):
        global img
        self.root = Tk()
        self.root.title('Credits')
        self.configureStyle()
        self.createLabels()
        self.orderLabels()

    def createLabels(self):
        self.lbl_created = Label(self.root, text='Created by:\n\tFleiner, Christian  \n\tGoergen, Konstantin  \n\tJohn, Felix '
                                                 '\n\tMichalczyk, Sven \n\tPickl, Max ')
        self.lbl_created.configure(background='black', foreground='green')

        self.lbl_opening = Label(self.root, text='Thanks to all Contributors\nof knowledge and source code')
        self.lbl_opening.configure(background='black', foreground='green')

        self.lbl_bk = HyperlinkLabel(self.root, "UC Berkeley CS188 Intro to AI")
        self.lbl_bk.setLink(r"http://ai.berkeley.edu/home.html")

        self.lbl_aifb = HyperlinkLabel(self.root, "Institut AIFB des KIT")
        self.lbl_aifb.setLink("http://www.aifb.kit.edu/web/Hauptseite")

    def configureStyle(self):
        self.root.configure(background='black')
        style = Style()
        style.configure("TLabel", background='black', foreground='green')

    def orderLabels(self):
        self.lbl_created.grid(row=1, padx=5, pady = 10)
        self.lbl_opening.grid(row=2, padx=5, pady = 10)
        self.lbl_bk.grid(row=3, padx=5)
        self.lbl_aifb.grid(row=4, padx=5)

    def show(self):
        self.root.mainloop()


class HyperlinkLabel(Label):
    link = ''

    def __init__(self, root, text):
        Label.__init__(self, master=root, text=text, cursor='hand2', background='black', foreground='blue')
        self.bind("<Button-1>", self.openLink)

    def setLink(self, str):
        self.link = str

    def openLink(self, event):
        webbrowser.open_new(self.link)

if __name__ == '__main__':
    app = MainWindow()
    app.configurestyles()
    app.createElements()
    app.orderElements()
    app.root.mainloop()


