import csv


def convert(infile, outfile):
    with open(outfile, 'w+', newline='', encoding='utf-8') as csvfile:
        # dialect就是定义一下文件的类型，我们定义为excel类型
        spamwriter = csv.writer(csvfile, dialect='excel')
        # 读要转换的txt文件，文件每行各词间以字符分隔
        with open(infile, 'r', encoding='utf-8') as filein:
            for line in filein:
                line_list = line.strip('\n').split('\t')  # 我这里的数据之间是以\t 间隔的
                line_list = line_list[::-1]  # 概要放在前面
                spamwriter.writerow(line_list)

if __name__ == '__main__':
  convert('data/train.txt', 'data/train.csv')
  convert('data/valid.txt', 'data/valid.csv')
  convert('data/test.txt', 'data/test.csv')
