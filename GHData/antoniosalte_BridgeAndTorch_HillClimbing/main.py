import sys
import os

from tkinter import *
from time import *

try:
    from tkinter import ttk
except:
    import ttk

from HillClimbing import *

entriesVelocity = []
labelsVelocity = []
persons = []


class MainRoot(Tk):

    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)

        container = Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (PageOne, PageTwo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("PageOne")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

    def create_persons(self):
        frame = self.frames["PageTwo"]
        frame.createPersons()


class PageOne(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.controller = controller

        labelError = Label(self, fg="red")

        labelInstruction = Label(self, text="Ingrese la cantidad de Personas y su Velocidades")
        labelInstruction.grid(row=0, column=0, columnspan=2)

        labelQuantity = Label(self, text="Cantidad Personas: ")
        labelQuantity.grid(row=1, column=0)

        entryQuantity = Entry(self)
        entryQuantity.grid(row=1, column=1)

        def getVelocity():
            persons.clear()
            for entry in entriesVelocity:
                persons.append(int(entry.get()))

        def init(*args):
            errorString = ""
            errorVelocity = ""
            for i in range(len(entriesVelocity)):
                entry = entriesVelocity[i]
                if not entry.get():
                    errorVelocity = errorVelocity + str(i + 1) + ","

            if errorVelocity != "":
                errorVelocity = errorVelocity[:-1]
                errorString = "Falta la velocidad: " + errorVelocity
            labelError["text"] = errorString

            if errorString == "":
                getVelocity()
                controller.show_frame("PageTwo")
                controller.create_persons()

        buttonNext = ttk.Button(self, text="Siguiente", command=init)

        def clearVelocity():
            for i in range(len(entriesVelocity)):
                entriesVelocity[i].destroy()
                labelsVelocity[i].destroy()
            entriesVelocity.clear()
            labelsVelocity.clear()

        def createVelocity(*args):
            clearVelocity()

            quantity = int(entryQuantity.get())
            entries = [Entry(self) for _ in range(quantity)]
            labels = [Label(self, text="Velocidad %s:" % (i + 1)) for i in range(quantity)]

            for i in range(quantity):
                rowIndex = 4 + i
                labels[i].grid(row=rowIndex, column=0)
                entries[i].grid(row=rowIndex, column=1)
                entriesVelocity.append(entries[i])
                labelsVelocity.append(labels[i])
            labelError.grid(row=(4 + quantity), column=0)

            buttonNext.grid(row=(5 + quantity), column=1)

        buttonCreate = ttk.Button(self, text="Crear", command=createVelocity)
        buttonCreate.grid(row=2, column=1)


class PageTwo(Frame):
    personsBox = []
    personsText = []
    labels = {}
    canvas = None
    buttonCreate = None

    def restart(self):
        python = sys.executable
        os.execl(python, python, * sys.argv)

    def end(self, total):
        extra = "s"
        if total <= 1:
            extra = ""

        labelEnd = Label(self, text="El menor tiempo para que todos cruzen el puente es de %s minuto%s" % (str(total), extra))
        labelEnd.pack()

        buttonRestart = ttk.Button(self, text="Reiniciar", command=self.restart)
        buttonRestart.pack()

    def frameSleep(self, sec):
        Frame.update(self)
        sleep(sec)

    def createLabels(self):
        labelValue = Label(self, text="Valor del Movimiento: 0")
        labelCost = Label(self, text="Tiempo Actual: 0")
        labelTotal = Label(self, text="Tiempo Total: 0")
        labelValue.pack()
        labelCost.pack()
        labelTotal.pack()

        self.labels["value"] = labelValue
        self.labels["cost"] = labelCost
        self.labels["total"] = labelTotal

    def changeText(self, label, _value):
        value = str(_value)
        text = ""
        if label == "value":
            text = "Valor del Movimiento: " + value
        elif label == "cost":
            text = "Costo Actual: " + value
        elif label == "total":
            text = "Costo Total: " + value

        self.labels[label]["text"] = text

    def createPersons(self):
        quantity = len(persons)
        extra = 0
        if quantity > 9:
            extra = (int((quantity - 9) / 3) * 30)
        height = 110 + extra
        _canvas = Canvas(self, width=350, height=height)

        _canvas.create_text(50, 10, text="Lado Inseguro")
        _canvas.create_text(300, 10, text="Lado Seguro")

        bridge = _canvas.create_rectangle(100, 25, 230, 85)
        _canvas.create_text(165, 95, text="Puente")
        _canvas.itemconfig(bridge, fill="#3d1600")

        for i in range(quantity):
            person = str(persons[i])
            x = 10 + ((i % 3) * 30)
            y = 20 + (int((i / 3)) * 30)
            box = _canvas.create_rectangle(x, y, x + 20, y + 20)
            text = _canvas.create_text(x + 10, y + 10, text=str(person))
            self.personsBox.append(box)
            self.personsText.append(text)
            _canvas.pack()
        self.canvas = _canvas
        self.createLabels()

    def movement(self, toMove, direction):
        self.frameSleep(1)
        pos = 100
        add = 10
        if direction == "left":
            pos = 200
            add = -10

        self.paint([max(toMove)], "yellow")

        for j in range(10):
            for i in range(len(toMove)):
                y = 30 + (i * 30)
                x = pos + (add * j)
                index = persons.index(toMove[i])
                box = self.personsBox[index]
                text = self.personsText[index]
                self.canvas.coords(box, x, y, x + 20, y + 20)
                self.canvas.coords(text, x + 10, y + 10)
            self.frameSleep(0.2)

        self.setPosition(toMove, direction)
        self.paint(toMove, "white")

    def setPosition(self, toSet, direction):
        add = 0
        if direction == "right":
            add = 240

        for _ in toSet:
            index = persons.index(_)
            x = add + (10 + ((index % 3) * 30))
            y = 20 + (int((index / 3)) * 30)

            box = self.personsBox[index]
            text = self.personsText[index]
            self.canvas.coords(box, x, y, x + 20, y + 20)
            self.canvas.coords(text, x + 10, y + 10)
            self.frameSleep(0.2)

    def paint(self, toPaint, color):
        for _ in toPaint:
            index = persons.index(_)
            box = self.personsBox[index]
            self.canvas.itemconfig(box, fill=color)
        self.frameSleep(0.3)

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.controller = controller

        def startAlgorithm():
            self.buttonCreate['state'] = 'disabled'
            hillClimbing(persons, self.movement, self.paint, self.changeText, self.end)

        _buttonCreate = ttk.Button(self, text="Empezar", command=startAlgorithm)
        _buttonCreate.pack()

        self.buttonCreate = _buttonCreate


if __name__ == "__main__":
    root = MainRoot()
    root.title("The Microsoft Problem")
    root.mainloop()