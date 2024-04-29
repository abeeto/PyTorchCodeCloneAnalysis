a = open('sample_text.py', 'r').read()
a = a[2:][:-1]

with open('text_clean.py', 'w') as f:
    f.writelines('\n')
    f.write(a)
    f.writelines('\n')
