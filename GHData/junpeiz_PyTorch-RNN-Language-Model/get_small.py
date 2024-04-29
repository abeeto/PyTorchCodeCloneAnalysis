"""
Get the small version of dataset to debug
"""

import os


for name in ['train', 'valid', 'test']:
    f_in = open(os.path.join('data', 'wikitext-2', name + '.txt'), 'r', encoding='utf8')
    f_out = open(os.path.join('data', 'wikitext-2', name + '_small.txt'), 'w', encoding='utf8')
    count = 0
    for line in f_in:
        f_out.write(line)
        count += 1
        if count > 10:
            break
    f_in.close()
    f_out.close()
print("Things finished")
