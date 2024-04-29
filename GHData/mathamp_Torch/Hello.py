import torch
import os


class H:
    _public_methods_ = ['h']
    _reg_progid_ = 'test64'
    _reg_clsid_ = '{54C36BD2-285A-4FB4-8CAC-D5D79F424579}'

    def h(self, text):
        print(text)
        return os.getcwd()


if __name__ == '__main__':
    print("Registering..")
    import win32com.server.register
    win32com.server.register.UseCommandLine(H)