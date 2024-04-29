a = open('sample_vision.py', 'r').read()
a = a[2:][:-1]

with open('vision_clean.py', 'w') as f:
    f.writelines('\n')
    f.write(a)
    f.writelines('\n')
