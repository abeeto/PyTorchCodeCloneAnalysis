import os
import zipfile
import gdown

def download():
    
    dir_name = 'tiny-imagenet-200'
    zip_file = 'tiny-imagenet-200.zip'
    
    if not os.path.isdir(dir_name): # if directory not present then download and unzip
        if not os.path.exists(zip_file):
            url = 'https://drive.google.com/uc?id=1n-jwJulLoPraTe7KImctFjhsvufi_6yq'
            gdown.download(url, output=zip_file, quiet=False)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(".")
            
if __name__ == '__main__':
    download()