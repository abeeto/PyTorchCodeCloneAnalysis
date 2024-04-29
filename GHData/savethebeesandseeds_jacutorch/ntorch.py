import sys
import os
import json
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", help="increase output verbosity",action="store_true")
    parser.add_argument("--symbolPath", default= "data.json", type = str, help="symbolPath to the symbol '.json' file")
    args = vars(parser.parse_args())
class SYMBOL:
    def __init__(self, path=None):
        self.symJsonPath = path
        self.name = None
        self.draw = None
        self.loadSymbol(self.symJsonPath)
        self.checkSymbolBuffer()
    def __str__(self):
        return str(self.__class__)+": "+str(self.__dict__)
    def loadSymbol(self, path=None):
        if(os.path.isfile(path) and os.path.splitext(path)[1]):
            with open(path, 'r') as f:
                self.__dict__.update(json.load(f))
            print("succesfully loaded symbol from <{}>".format(os.path.join(os.path.abspath(os.getcwd()), path)))
            if(args["verbose"]):
                print("<<Extracted content>>: {}".format(self))
        else:
            assert os.path.exists(path),"{} does not exist".format(os.path.join(os.path.abspath(os.getcwd()), path))
            assert not(os.path.isdir(path)),"{} is a directory not a '.json' file.".format(path)
            assert os.path.isfile(path),"{} is not a valid path.".format(path)
            assert os.path.splitext(path)[1]=='.json', "{} is not valid. Must be '.json' format.".format(path)
    def checkSymbolBuffer(self):
        if(not(self.name)):
            print("waka: {}".format(self.name))
            aux = "<symbolError> symbol {} must have a name.\n".format(self.symJsonPath) + r'<Help>, try {"name":"foo"...}, to define foo as the symbol name.'
            assert self.name, aux
        if(not(self.draw)): 
            aux = "<symbolError> symbol {} must have a defined graphical symbol a.k.a. its 'draw'.\n".format(self.symJsonPath) + r'<Help>, try {..."draw":"\\mathcal{N}"...}, to define a symbol graphic based on Latex syntax.'
            assert self.draw, aux
if __name__ == '__main__':# Main
    if(args["verbose"]):
        print("<scrip arguments>: {}".format(args))
    sym_a = SYMBOL(path=args["symbolPath"])
    print("CLASS:", sym_a)


