import os
import glob
import pandas
import shutil


print("Starting")

def concatenate(indir="C:\\Users\\Shubham\\Documents\\Study\\mastersCourse\\Project\\Dataset\\electricity_price\\Extracted_Dataset",
                outFile="C:\\Users\\Shubham\\Documents\\Study\\mastersCourse\\Project\\Dataset\\electricity_price\\Extracted_Dataset\\Concatenated_price.csv"):

    os.chdir(indir)
    fileList = glob.glob("*.csv")

    print("Starting extraction")

    with open(outFile,'wb') as outfile:
        for i, fname in enumerate(fileList):
            with open(fname, 'rb') as infile:

                if i != 0:
                    infile.readline() #throw away header on all but first file

                shutil.copyfileobj(infile,outfile) #block copy rest of file from input to output without parsing


    print("Concateonation done")


concatenate()