import json
import codecs
import subprocess
import re

dataset_path = '/Users/alvin/NCC/git_repository/multi30k/data/task1/raw/raw_data/'
de_path_train = dataset_path + 'train.de'
de_path_val = dataset_path + 'val.de'
de_path_test = dataset_path + 'test_2016_flickr.de'
en_path_train = dataset_path + 'train.en'
en_path_val = dataset_path + 'val.en'
en_path_test = dataset_path + 'test_2016_flickr.en'

def read_text(filename):
    fc = subprocess.getoutput('file -b --mime-encoding %s' %filename)
    with codecs.open(filename, 'r', encoding=fc) as f:
        #content = f.readlines()
        content = f.read().splitlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    #content = [x.strip() for x in content]
    content = [re.sub(r'\"\"','\"',x) for x in content] # train_list[7365] bug: dobule quota in front-end
    content = [re.sub(r'\t','',x) for x in content]     # train_list[7365] bug: specail character \t
    content = [re.sub(r'\s+',' ',x) for x in content]   # waring 
    content = [re.sub(r'\"','\'',x) for x in content]   # double quota is special char, traing_list have 147 samples  
    return content

train_de = read_text(de_path_train)
train_en = read_text(en_path_train)
val_de = read_text(de_path_val)
val_en = read_text(en_path_val)
test_de = read_text(de_path_test)
test_en = read_text(en_path_test)

def update_string(orig_string, fix_length=20):
    ori_str_list = orig_string.split(' ')
    new_str_list = list(map(lambda x: ori_str_list[x] if x < len(ori_str_list) else '<apad>', range(fix_length)))
    new_string = ' '.join(new_str_list)
    return new_string

fix_length = 10
if fix_length != 0:
    train_de = list(map(lambda x: update_string(x,fix_length), train_de)) 
    train_en = list(map(lambda x: update_string(x,fix_length), train_en))
    val_de   = list(map(lambda x: update_string(x,fix_length), val_de))  
    val_en   = list(map(lambda x: update_string(x,fix_length), val_en))
    test_de  = list(map(lambda x: update_string(x,fix_length), test_de))    
    test_en  = list(map(lambda x: update_string(x,fix_length), test_en))    

train_list = list(map(lambda x,y: '{"DE": "'+ x +'", "EN": "'+y +'"}',train_de, train_en))
valid_list = list(map(lambda x,y: '{"DE": "'+ x +'", "EN": "'+y +'"}',val_de, val_en))
test_list = list(map(lambda x,y: '{"DE": "'+ x +'", "EN": "'+y +'"}',test_de, test_en))

with codecs.open('train10.json', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_list))

with codecs.open('valid10.json', 'w', encoding='utf-8') as f:
    f.write('\n'.join(valid_list))

with codecs.open('test10.json', 'w', encoding='utf-8') as f:
    f.write('\n'.join(test_list))

'''
train_list = list(map(lambda x,y: {'DE': x , 'EN': y },train_de, train_en))
val_list = list(map(lambda x,y: {'DE': x , 'EN': y },val_de, val_en))
test_list = list(map(lambda x,y: {'DE': x , 'EN': y },test_de, test_en))

with codecs.open('train.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(train_list,f,indent=4,ensure_ascii=False))

with codecs.open('valid.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(valid,f,indent=4,ensure_ascii=False))

with codecs.open('test.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(test_list,f,indent=4,ensure_ascii=False))
'''

# json format check
err_count = 0
for i in range(len(train_list)):
    try:
        aa = json.loads(train_list[i])
    except:
        err_count += 1
        print(str(i) + ' error ==> \t' + train_list[i])
print('err_count = ' + str(err_count))