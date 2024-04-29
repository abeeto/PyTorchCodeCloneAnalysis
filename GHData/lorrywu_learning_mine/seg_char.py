# coding:utf-8
import jieba
import json
import re
import prettyprint
from collections import defaultdict

user_dict_path = 'data/users'


if __name__ == '__main__':

    with open(user_dict_path, 'r') as fp:
        for id in fp:
            id = id.strip()
            content = defaultdict(list)
            fin = open('data/' + str(id), 'r')
            for line in fin:
                line = line.replace('转发微博','')
                string = line.decode("utf-8")

                filtrate = re.compile(u'[^\u4E00-\u9FA5]')#非中文
                filtered_str = filtrate.sub(r'', string)#replace

                words = list(filtered_str)



                words = [word.encode('utf-8') for word in words]
                new_list = [x for x in words if x != '']
                print prettyprint.pp_str(words)

                if len(new_list) > 0:
                    content[id].append(new_list)
            if len(content[id]) > 0:
                fo = open('seg_char_data/' + str(id) + '.seg', 'w')
                for line in content[id]:
                    json.dump(line, fo, ensure_ascii=False)
                    fo.write('\n')
                content.clear()
