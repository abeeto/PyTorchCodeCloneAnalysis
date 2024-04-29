import json


def convert(infile, outfile):
  fin = open(infile, 'r', encoding='utf-8')
  fout = open(outfile, 'w', encoding='utf-8')
  for line in fin:
    # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
    # 注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
    document, summary = line.strip('\n').split('\t')
    # json.dumps把python对象(诸如dict/list/tuple/string等数据结构)转换为字符串。只不过这个字符串比较特别，其特别之处在于它的语法格式同json保持一致。如将字典中的单引号均置为双引号
    # 写入fout文件
    print(json.dumps({'document': document, 'summary': summary}), file=fout)
  fin.close()
  fout.close()

if __name__ == '__main__':
  convert('data/train.txt', 'data/good_format_train.txt')
  convert('data/valid.txt', 'data/good_format_valid.txt')
  convert('data/test.txt', 'data/good_format_test.txt')
