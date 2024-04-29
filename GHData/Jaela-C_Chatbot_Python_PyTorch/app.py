from tkinter import *
from chat import get_response, bot_name

BG_PRIMARY = "#707E8B"
BG_SECONDARY = "#969FF9"
BG_COLOR = "#394959"
TEXT_COLOR = "#EAECEE"

FONT = "Roboto 14"
FONT_BOLD = "Roboto 16 bold"

class ChatBotApplication:
    
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("ChatBot")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BG_COLOR)

        #head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR, text="Bienvenidos", font=FONT_BOLD, pady=15)
        head_label.place(relwidth=1)

        #divider
        line = Label(self.window, width=550, bg=BG_PRIMARY)
        line.place(relwidth=1, rely=0.07, relheight=0.12)

        #text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        #scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        #bottom label
        bottom_label = Label(self.window, bg=BG_PRIMARY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        #message box
        self.message_box = Entry(bottom_label, bg="#394959", fg=TEXT_COLOR, font=FONT)
        self.message_box.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.message_box.focus()
        self.message_box.bind("<Return>", self._on_enter_press)

        #send button
        send_button = Button(bottom_label, text="Enviar", fg=TEXT_COLOR, font=FONT, width=20, bg=BG_PRIMARY, command=lambda: self._on_enter_press(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def _on_enter_press(self, event):
        message = self.message_box.get()
        self._insert_message(message, "User")

    def _insert_message(self, message, sender):
        if not message:
            return
        
        self.message_box.delete(0, END)
        messageUser = f"{sender}: {message}\n\n"
        self.text_widget.configure(cursor="arrow", state=NORMAL)
        self.text_widget.insert(END, messageUser)
        self.text_widget.configure(state=DISABLED)

        messageBot = f"{bot_name}: {get_response(message)}\n\n"
        self.text_widget.configure(cursor="arrow", state=NORMAL)
        self.text_widget.insert(END, messageBot)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)

if __name__ == "__main__":
    app = ChatBotApplication()
    app.run()