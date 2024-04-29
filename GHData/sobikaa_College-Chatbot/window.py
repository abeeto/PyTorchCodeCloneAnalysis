from tkinter import *
from chat import get_response

bot_name = "Sunflower"


class ChatbotWindow:

    def __init__(self):
        self.writeMessage = None
        self.messageWindow = None
        self.chatBox = Tk()
        self.main_window()

    def run(self):
        self.chatBox.mainloop()

    def main_window(self):

        self.chatBox.title("Ask Sunflower")
        self.chatBox.configure(width=470, height=550, bg = "black")
        self.chatBox.resizable(width=False, height=False)

        self.messageWindow = Text(self.chatBox, bg="black", width=20, height=2, fg="white", padx=5, pady=5)
        self.messageWindow.place(relheight=0.845, relwidth=1)
        self.messageWindow.configure(cursor="arrow", state=DISABLED)

        scrollbar = Scrollbar(self.messageWindow)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.messageWindow.yview)

        textbox = Label(self.chatBox, bg="gray", height=70)
        textbox.place(relwidth=1, rely=0.85)

        self.writeMessage = Entry(textbox, bg = "white")
        self.writeMessage.place(relwidth=0.97, relheight=0.05, rely=0.007, relx=0.016)
        self.writeMessage.focus()
        self.writeMessage.bind("<Return>", self.receive_message)

    def receive_message(self, event):
        message = self.writeMessage.get()
        self.send_message(message, "You")

    def send_message(self, message, sender):
        if not message:
            return

        self.writeMessage.delete(0, END)

        message1 = f"{sender}: {message}\n\n"

        self.messageWindow.configure(state=NORMAL)
        self.messageWindow.insert(END, message1)
        self.messageWindow.configure(state=DISABLED)

        message2 = f"{bot_name}: {get_response(message)}\n\n"

        self.messageWindow.configure(state=NORMAL)
        self.messageWindow.insert(END, message2)
        self.messageWindow.configure(state=DISABLED)
        self.messageWindow.see(END)


if __name__ == "__main__":
    app = ChatbotWindow()
    app.run()

