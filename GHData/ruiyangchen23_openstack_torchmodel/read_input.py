import numpy as np
from torchvision import transforms, utils
import urllib.request as urllib

from cinderclient import client 


def download_data_external(URL,data_format):
    """  This function is used to download external data into the local machine
    
    Args:
        URL(str): the url that points the external data. 
        data_format(str): the format user specified during boot time, needed for uncompress. 

    Returns:
        None, the file will be downloaded to local disk
    """
    connector = urllib.URLopener()
    if data_format=='gz':
        connector.retrieve(URL,'./cvdata.gz')
    elif data_format=='zip':
        connector.retrieve(URL,'./cvdata.zip')
    elif data_format=='tar':
        connector.retrieve(URL,'./cvdata.tar')
    elif data_format =='uncompressed':
        connector.retrieve(URL,'./cvdata/')

def download_data_internal():   
    """This function is used to download data from openstack volume storage 
    
    Args:

    Returns:
        None, the file will be downloaded to local disk
    """
    return 