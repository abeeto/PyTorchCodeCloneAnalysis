import glob
import os
import re

here = lambda *x: os.path.abspath(os.path.join(os.path.dirname(__file__), *x))
reuters_dir = here('reuters21578', )
data_dir = here('data', )


if not os.path.exists(data_dir):
    os.makedirs(data_dir)
os.chdir(reuters_dir)
for file in glob.glob('*.sgm'):
    current_file = os.path.join(reuters_dir, file)
    print 'Removing special characters from file %s' % current_file
    with open(current_file, 'r') as f:
        read_data = f.read()
        # 00 - 47; 123 - 159 punctuation unused
        clean_data = re.sub(r'&#((\d|[0-3]\d)|(4[0-7]));', '', read_data)
        clean_data = re.sub(r'&#(1(2[3-9]|[3-4]\d|5[0-9]));', '', clean_data)
        # < less-than sign, it will be removed anyway
        clean_data = re.sub(r'&lt;', '', clean_data)
        # remove f****reute that appears in unknown tag
        clean_data = re.sub(r'f(\d{,4})reute', '', clean_data)
        clean_data = re.sub(r'-', ' - ', clean_data)
        clean_data = re.sub(r'><', '> <', clean_data)
        new_file = os.path.join(data_dir, file)
        with open(new_file, 'w+') as nf:
            nf.write(clean_data)
