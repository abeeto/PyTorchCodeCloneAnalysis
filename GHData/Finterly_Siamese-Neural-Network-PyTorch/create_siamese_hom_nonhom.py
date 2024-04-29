"""
Created on 07 Mar 2019

Note:gene3d missing 3.40.50.150*3.40.390.10*3.40.390.10

@author: Finterly
"""
from collections import defaultdict
import csv
import random
import sys
import time
import os

db = 'supfam'
diri = os.path.join(os.path.expanduser('~'), 'thesis', 'thesis_eclipse','data','csv','csv_siamese')
siamese_file =os.path.join(diri, 'special2', "siamese_file_" + db+ "_nomax50_hom.csv")
siamese_file_2 =os.path.join(diri, 'special2', "siamese_file_" + db+ "_nomax50_nonhom.csv")
hom_file = os.path.join(diri, 'benchmark', db + "_hom-pairs_nomax50.txt")
nonhom_file = os.path.join(diri, 'benchmark', db + "_nonhom-pairs_nomax50.txt")
fasta_file ="C:/Users/Finterly/thesis/thesis_eclipse/data/fastas/pairdata_all.fasta" #MISSING SOME FASTAS!!!!
seq_not_found = os.path.join(diri, 'special2', db+"_seq_not_found_siamese.csv")
missing_fasta = os.path.join(diri, 'special2', db+"_missing_fasta.txt")

def create_empty_csv (file_path):
    with open(file_path, 'wb') as my_empty_csv:
        print("csv file created")  

def retrieve_seq(uid, fasta_file):
    try:
        with open(fasta_file) as f:
            if next((l for l in f if uid in l), None).strip():
                seq = next(f).strip()
                return seq
    except:
        return False 

def seq_to_num(seq):
    amino_acid_to_idx = {'0': 0, 'A' :1, 'R':2, 'N':3, 'D':4, 'C':5, 'Q':6, 'E':7, 'G':8, 'H':9, 'I':10, 'L':11, 'K':12, 'M':13, 'F':14, 'P':15, 'S':16, 'T':17, 'W':18, 'Y':19, 'V':20, 'B':21, 'Z':22, 'X':23, 'J':24, 'U':25}  
    seq_list = ([amino_acid_to_idx[i] for i in seq])
    num_seq = ''.join(str(x) for x in seq_list)
    return num_seq

def create_siamese_csv (hom_file, nonhom_file,siamese_file, fasta_file, seq_not_found): 
    missing = set()
    with open(seq_not_found, 'w',1) as snf: 
        with open(siamese_file, 'w', 20000) as out_f:
            with open(hom_file) as hom_f:
                for line in hom_f: 
                    motif = line.strip().split(":")[1].split("@")[0] 
                    uid1 = line.strip().split("@")[1].split("_")[0] 
                    uid2 =  line.strip().split("_")[1]
                    seq1 = retrieve_seq(uid1, fasta_file)
                    seq2 = retrieve_seq(uid2, fasta_file)
                    if seq1 == False:
                        missing.add(uid1)
                    elif seq2== False:
                        missing.add(uid2)
                        snf.write(line) 
                    else:                                               
                        out_f.write("0," + seq1 +"," + seq2+","+ motif +"," +motif+ "\n")
    #                 out_f.write("0," + seq1 +("0" *pad_len)+"," + seq2+("0" *pad_len)+","+motif +"," +motif+ "\n")
            
            
def create_siamese_csv_2 (hom_file, nonhom_file,siamese_file, fasta_file, seq_not_found): 
    missing = set()
    with open(seq_not_found, 'w',1) as snf: 
        with open(siamese_file, 'w', 20000) as out_f:            
            with open(nonhom_file) as nonhom_f:
                for line in nonhom_f:   
                    part1 = line.strip().split("@")[0]
                    part2 = line.strip().split("@")[1]
                    motif1 = part1.strip().split(":")[1].split("_")[0] 
                    uid1 = part1.strip().split("_")[1]
                    motif2 = part2.strip().split("_")[0]
                    uid2 =  part2.strip().split("_")[1]
                    seq1 = retrieve_seq(uid1, fasta_file)
                    seq2 = retrieve_seq(uid2, fasta_file)
                    if seq1 == False:
                        missing.add(uid1)
                        snf.write(line) 
                    elif seq2== False:
                        missing.add(uid2)
                        snf.write(line) 
                    else:        
                        out_f.write("1," + seq1 +"," + seq2+","+motif1 +"," +motif2+ "\n")
    #                 out_f.write("0," + seq1 +("0" *pad_len)+"," + seq2+("0" *pad_len)+","+motif1 +"," +motif2+ "\n")
    with open(missing_fasta, 'w') as mis: 
        for uid in missing:
            mis.write(uid + '\n')
            
#create_empty_csv(siamese_file)
#create_empty_csv(seq_not_found)
create_siamese_csv(hom_file, nonhom_file,siamese_file, fasta_file, seq_not_found)
create_siamese_csv_2(hom_file, nonhom_file,siamese_file_2, fasta_file, seq_not_found)





               