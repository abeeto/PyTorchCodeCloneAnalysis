#!/usr/bin/env python

import os

host=os.uname()[1]
filename=f'{host}.txt'

# with open(filename,'w') as ofile:
#     ofile.write(f'Host: {os.uname()[1]}')

print(f'Host: {host}')
