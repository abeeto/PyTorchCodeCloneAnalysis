import codecs
import collections
from operator import itemgetter
#首先按照词频顺序为每个词汇分配一个编号，然后将词汇表保存到一个独立的vocab文件中。

def deal(lang):
    # 训练集数据文件
    ROOT_PATH = "./summary1860/"
    if lang == "zh":
        RAW_DATA = ROOT_PATH + "TED2013.zh"
        # 输出的词汇表文件
        VOCAB_OUTPUT = ROOT_PATH + "zh.vocab"
        # 中文词汇表单词个数
        VOCAB_SIZE = 4000
    elif lang == "en":
        RAW_DATA = ROOT_PATH + "TED2013.en"
        VOCAB_OUTPUT = ROOT_PATH + "en.vocab"
        VOCAB_SIZE = 10000
    else:
        print("what?")

    # 统计单词出现的频率
    counter = collections.Counter()
    with codecs.open(RAW_DATA, "r", "utf-8") as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1

    # 按照词频顺序对单词进行排序
    sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
    print(sorted_word_to_cnt)
    sorted_words = [x[0] for x in sorted_word_to_cnt]

    # 在后面处理机器翻译数据时，除了"<eos>"，还需要将"<unk>"和句子起始符"<sos>"加入
    # 词汇表，并从词汇表中删除低频词汇。在PTB数据中，因为输入数据已经将低频词汇替换成了
    # "<unk>"，因此不需要这一步骤。
    sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words
    if len(sorted_words) > VOCAB_SIZE:
        sorted_words = sorted_words[:VOCAB_SIZE]

    with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
        for word in sorted_words:
            file_output.write(word + "\n")

def to_vocab():
    lang = ["zh", "en"]
    for i in lang:
        deal(i)

def word_to_num(lang):
    # 训练集数据文件
    ROOT_PATH = "./summary1860/"
    if lang == "zh":
        # 原始的训练集数据文件
        RAW_DATA = ROOT_PATH + "TED2013.zh"
        # 上面生成的词汇表文件
        VOCAB = ROOT_PATH + "zh.vocab"
        # 将单词替换成为单词编号后的输出文件
        OUTPUT_DATA = ROOT_PATH + "zh.number"
    elif lang == "en":
        RAW_DATA = ROOT_PATH + "TED2013.en"
        VOCAB = ROOT_PATH + "en.vocab"
        OUTPUT_DATA = ROOT_PATH + "en.number"
    else:
        print("what?")
    # 读取词汇表，并建立词汇到单词编号的映射。
    with codecs.open(VOCAB, "r", "utf-8") as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]
    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}
    # 如果出现了被删除的低频词，则替换为"<unk>"。
    def get_id(word):
        return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]
    fin = codecs.open(RAW_DATA, "r", "utf-8")
    fout = codecs.open(OUTPUT_DATA, 'w', 'utf-8')
    for line in fin:
        # 读取单词并添加<eos>结束符
        words = line.strip().split() + ["<eos>"]
        # 将每个单词替换为词汇表中的编号
        out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
        fout.write(out_line)
    fin.close()
    fout.close()

def to_num():
    lang = ["zh", "en"]
    for i in lang:
        word_to_num(i)

if __name__ == "__main__":
    # 处理的语言
    to_vocab()
    to_num()