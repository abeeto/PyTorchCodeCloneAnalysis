
import socket,curses
import threading,binascii
HOST=input('enter Host (default : 127.0.0.1) ')
if HOST=='' or HOST.isspace==True:
    HOST="127.0.0.1"
    
PORT=input('(default-10001) PORT : ')
if PORT=='' or PORT.isspace==True:
    PORT=10001
else:
    PORT=int(PORT)

BUFFER_SIZE = 1024

class ChatClient:
    '''
    class for chat client throw which we communicate with the server
    '''
    def __init__(self, host, port):
        #create socket, connect to server
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))


    def send_message(self, message):
        '''
        function for sending message to server
        '''
    
        self.sock.sendall(binascii.b2a_uu( message.encode()))
    
    def receive_message(self):
        '''
        function for receiving message from server
        '''
        data = self.sock.recv(BUFFER_SIZE)
        if not data:
            return None
        data = binascii.a2b_uu(data).decode()
        return data

class ChatInterface:
    '''
    class for creating required console interface
    '''
    def __init__(self):
        #set need variables
        self.KEYS_IGNORE = [
                curses.KEY_LEFT,
                curses.KEY_RIGHT
            ]
        self.MAX_VIEW = 19
        self.MAX_LENGTH = 76
        self.current_str = 0
        
        self.view_y = 2
        self.input_y = 2
        
        #lock to work with sockets
        self.lock = threading.Lock()
        
        #create client connection
        self.client_conn = ChatClient(HOST, PORT)
        
        #create console interface
        self.init_TUI()  
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_WHITE, -1)
    
    def init_TUI(self):
        '''
        function for create chat "face" 
        '''
        self.main_win = curses.initscr()
        curses.noecho()
        curses.cbreak()
     
        self.main_win.clear()
        
        #all records in view block
        self.all_records = []
        self.title = "Enter your name:                  made by security softwares"
        
        self.all_records.append((self.title, 3))
        
        #view block
        self.chat_view = curses.newwin(20, 80, 0, 0)
       
        self.chat_view.addstr(1, 1, self.title)
        self.chat_view.border()
        self.chat_view.keypad(1)
        self.chat_view.refresh()

        #user input block
        self.chat_input = curses.newwin(4, 80, 20, 0)
        self.init_chat_input(self.chat_input)

    def init_chat_input(self, win):
        '''
        function for initialization input block
        '''
        self.clear_win(win)
        win.keypad(1)
        win.move(1, 1)

    def refresh_chat_input(self, all_records):
        '''
        function to go to new line in user input block
        '''
        self.clear_win(self.chat_input)
        self.chat_input.addstr(1, 1, all_records[-1])
        self.chat_input.move(2, 1)
 
    def main(self):
        '''
        function to create receive and send loop
        '''
        tr1 = threading.Thread(target=self.receive_loop)
        tr1.daemon = True
        tr1.start()
        self.main_loop()        
        
    def receive_loop(self):
        '''
        function for receive loop. Receive message from server
        and display it.
        '''
        while True:
            data = self.client_conn.receive_message()
            if not data: break
            strings = self.separate_string(data)
            
            #thread block
            self.lock.acquire()
            cur_y, cur_x = self.chat_input.getyx()
            self.add_new_records(strings, 2)
            
            self.chat_view.refresh()
            self.chat_input.move(cur_y, cur_x)
            self.chat_input.refresh()
            #thread unblock
            self.lock.release()
            
            
    def add_new_records(self, user_strings, color_pair):
        '''
        function for add new records and display them
        '''
        for line in user_strings:
            self.all_records.append((line, color_pair))
            if self.view_y == self.MAX_VIEW:
                self.refresh_chat_view(self.all_records)
            else:
                self.chat_view.addstr(
                                    self.view_y, 
                                    1, 
                                    line, 
                                    curses.color_pair(color_pair)
                                    )
                self.view_y += 1
            self.current_str = len(self.all_records)
    
    def separate_string(self, string):
        '''
        function for separate long string
        '''
        ls = []
        if len(string) > self.MAX_LENGTH:
            while string:
                ls.append(string[:self.MAX_LENGTH])
                string = string[self.MAX_LENGTH:]
        else:
            ls.append(string)
        return ls
        
        
    def main_loop(self):
        '''
        function for create sending loop
        '''
        while True:
            user_strings = self.handle_user_input()    
            
            if user_strings[0]:
                #thread block
                self.lock.acquire()
                user_strings[0] = ">>> " + user_strings[0]
                new_strs = self.separate_string(''.join(user_strings))        
                self.add_new_records(new_strs, 1)
                
                user_strings[0] = user_strings[0][4:]
                
                self.chat_view.refresh()
                self.init_chat_input(self.chat_input)
                #thread unblock
                self.lock.release()
                #send message to server
                self.client_conn.send_message(''.join(user_strings))
    
    def handle_user_input(self):
        '''
        function for handling user input. Return list of strings 
        separated by MAX_LENGTH
        '''
        user_strings = []
        user_str = ""
        while True:
            try:
                #wait input
                k = self.chat_input.get_wch()
                
                #delete handle
                if k == curses.KEY_BACKSPACE:
                    if len(user_str) > 0:
                        user_str = user_str[:-1]
                        cur_y, cur_x = self.chat_input.getyx()
                        self.clear_win(self.chat_input)
                        if cur_y != 1:
                            self.chat_input.addstr(
                                                cur_y-1, 1, 
                                                user_strings[-1]
                                                )
                        self.chat_input.addstr(cur_y, 1, user_str)
                        self.chat_input.refresh()
                    else:
                        self.clear_win(self.chat_input)
                        user_str = self._input_up_str(user_strings)
                        continue
                elif k not in self.KEYS_IGNORE:
                    #mouse scroll down
                    if k == curses.KEY_DOWN:
                        cur_y, cur_x = self.chat_input.getyx()
                        self.backward_chat_view(cur_y, cur_x)            
                        continue
                    #mouse scroll up
                    elif k == curses.KEY_UP:
                        cur_y, cur_x = self.chat_input.getyx()
                        self.forward_chat_view(cur_y, cur_x)
                        continue
                    #handle long string
                    if len(user_str) > self.MAX_LENGTH:
                        user_strings.append(user_str)
                        user_str = ""
                        self.refresh_chat_input(user_strings)
                    #end user input
                    if k == '\n':
                        user_strings.append(user_str)
                        break
                    try:
                        user_str += k
                        self.chat_input.addch(k)
                    except TypeError:
                        pass
            except curses.error:
                pass
        return user_strings
        
    def refresh_chat_view(self, all_records):
        '''
        function for update view block if records go beyond the block
        '''
        n = len(all_records)
        if n >= self.MAX_VIEW:
            self.clear_win(self.chat_view)
            y = 1
            n = len(all_records)
            for i in range(self.MAX_VIEW-1, 0, -1):    
                self.chat_view.addstr(
                                    y, 
                                    1, 
                                    all_records[n-i][0], 
                                    curses.color_pair(
                                            all_records[n-i][1]
                                            )
                                    )
                y += 1
    
    
    def backward_chat_view(self, cur_y, cur_x):
        '''
        function for update view block when scrolling to down
        '''
        n = len(self.all_records)
        if n >= self.MAX_VIEW and self.current_str <= n:
            self.clear_win(self.chat_view)
            #update view block
            self._update_view_block()
            if self.current_str != n:
                self.current_str += 1
            self.chat_view.refresh()
        self.chat_input.move(cur_y, cur_x)
        self.chat_input.refresh()
    
    def forward_chat_view(self, cur_y, cur_x):
        '''
        function for update view block when scrolling to up
        '''
        n = len(self.all_records)
        if n >= self.MAX_VIEW and self.current_str >= self.MAX_VIEW-1:
            self.clear_win(self.chat_view)
            #update view block
            self._update_view_block()
            self.current_str -= 1
            self.chat_view.refresh()
        self.chat_input.move(cur_y, cur_x)
        self.chat_input.refresh()
        
    def _update_view_block(self):
        y = 1
        for i in range(self.MAX_VIEW-1, 0, -1):    
            j = self.current_str - i
            if j >= 0:
                self.chat_view.addstr(
                                    y,
                                    1,
                                    self.all_records[j][0], 
                                    curses.color_pair(
                                            self.all_records[j][1]
                                            )
                                    )
            else:
                break
            y += 1
            
    def clear_win(self, win):
        '''
        function for clear block
        '''
        win.clear()
        win.border()
        win.refresh()    

    def _input_up_str(self, user_strings):
        '''
        function for update user input block 
        when delete all char in sting and go up to next string
        '''
        new_str = ""
        if user_strings:
            if len(user_strings) == 1:
                self.chat_input.addstr(1, 1, user_strings[0])
                new_str = user_strings.pop()
            else:
                self.chat_input.addstr(1, 1, user_strings[-2])
                self.chat_input.addstr(2, 1, user_strings[-1])
                new_str = user_strings.pop()
                
        else:
            self.chat_input.move(1, 1)
        self.chat_input.refresh()
        return new_str
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, tb):
        curses.endwin()
        

with ChatInterface() as client:
    client.main()
    


