import pandas as pd
from collections import Counter
import os
import shutil

# f=pd.read_csv('filenames.txt', header=None)
# f.columns=['filenames']
# f[:5]

# fn=f['filenames'].values
# print('total ',len(fn))
# fn[:5]
dtd='bi_concepts1553'
cwd=os.getcwd()
fn=os.listdir(os.path.join(cwd, dtd))
print('initially ',len(fn))

ad=[x.split('_')[0] for x in fn]

add=Counter(ad)

# len(add.keys())

fad=[]
[fad.append(k) for k,v in add.items() if v<3]
os.chdir(dtd)

print(fad) 

for fa in fad:
	for f in fn:  
		if fa in f.split()[0]:
			shutil.rmtree(f)
print('going to remove directories ',len(fad))
di=os.listdir(os.getcwd())
print('now left ', len(di))

# no=[x.split('_')[1] for x in fn]
# no.sort()
# print(len(set(no)))
# set(no)
# di

for i in di: dl=len(os.listdir(os.path.join(os.getcwd(),i))); shutil.rmtree(i) if dl<501 else 0
di=os.listdir(os.getcwd())
print('now left ', len(di))