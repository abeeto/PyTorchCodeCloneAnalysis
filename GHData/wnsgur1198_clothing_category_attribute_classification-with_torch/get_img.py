# -*- coding: UTF-8 -*-
import urllib.request
#from main import Analyze
from demo.test_cate_attr_predictor import main

class GetImage:
    def __init__(self):
        #print('get image')
        pass

    def getImage(self, url):
        #anl = Analyze()
        
        #urllib.request.urlretrieve(url, "images/0.jpg")
        urllib.request.urlretrieve(url, "demo/imgs/0.jpg")
        #return anl.analyze()
        return main() 


