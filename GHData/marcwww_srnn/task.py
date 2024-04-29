import numpy as np
import random

np.random.seed(1234)
random.seed(1234)
def reverse(max_int,sequence_length):
    sequence = np.random.randint(max_int,size=sequence_length).astype(np.int8)
    input_sequence  = -1 * np.ones((2 * sequence_length + 1,),dtype=np.int8)
    output_sequence = -1 * np.ones((2 * sequence_length + 1,),dtype=np.int8)
    input_sequence[sequence_length] = -2
    input_sequence[:sequence_length] = sequence
    output_sequence[-sequence_length:] = sequence[::-1]
    return input_sequence,output_sequence

length=np.random.randint(64-8)+8
for _ in range(10):
    x,y=reverse(128,length)
    print(x,y)
