# coding: utf-8

import codecs

from logger import Logger

def to_np(x):
    return x.data.cpu().numpy()

class Vocab(object):
    def __init__(self, vocfile, unk_id, pad_id):
        self._word2id = {}
        self._id2word = {}
        self.unk_id = unk_id
        self.padding_id = pad_id
        self._voc_name = vocfile
        with codecs.open(vocfile, mode='rb', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) != 2:
                        print 'illegal voc line %s' % line
                        continue
                    id = int(parts[1])
                    self._word2id[parts[0]] = id
                    self._id2word[id] = parts[0]
        self._vocab_size = max(self._word2id.values()) + 1
        self.unk = self._id2word[self.unk_id]
        self.PADDING = self._id2word[self.padding_id]
        if self._vocab_size != len(self._word2id):
            print 'in vocab file {}, vocab_max {} not equal to vocab count {}, maybe empty id or others' \
                .format(vocfile, self._vocab_size, len(self._word2id))

    def __str__(self):
        return self._voc_name

    def getVocSize(self):
        return self._vocab_size

    def getWord(self, id):
        return self._id2word[id] if self._id2word.has_key(id) else self.unk

    def getID(self, word):
        return self._word2id[word] if self._word2id.has_key(word) else self.unk_id

    @property
    def word2id(self):
        return self._word2id

    @property
    def id2word(self):
        return self._id2word

class Config(object):
    def __init__(self):
        self.dic = {}

    def __setitem__(self, key, value):
        # if key == 'EmbSizes':
        #     print '\n<<<<<', 'EmbSizes'
        self.dic[key] = value

    def __getitem__(self, item):
        # if item == 'EmbFiles':
        #     print '\n<<<', 'EmbFiles'
        # if item == 'EmbSizes':
        #     print '\n<<<<<', 'EmbSizes'
        return self.dic[item]


def get_conf_base():
    config = {}
    config['output_dim'] = 300
    config['early_stop'] = 10
    config['weight_scale'] = 0.05
    config['max_seq_length'] = 4
    config['max_label_length'] = 100
    config['batch_size'] = 32  #24
    config['hidden_size'] = 400
    config['decoder_layers'] = 1
    config['encoder_filter_num'] = 400
    config['encoder_outputs_size'] = config['hidden_size']
    config['decoder_output_size'] = 28
    config['clip_norm'] = 1  # 5.0
    config['beam_size'] = 8
    config['EOS_token'] = 2
    config['PAD_token'] = 0
    config['UNK_token'] = 1
    config['X'] = 17
    config['att_mode'] = 'general'
    # config['OutTags'] = Vocab('res/ner_xx', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    # config['BioOutTags'] = Vocab('res/ner_bio.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    # config['WordId'] = Vocab('res/voc.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['filter_size'] = 3
    config['save_freq'] = 30
    config['dropout'] = 0.25  # 0.25
    config['multi_cuda'] = [1, 3]
    config['use_multi'] = False

    # config['max_char'] = 25
    # config['char_emb_dim'] = 50
    # config['CharVoc'] = Vocab('res/char.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    # config['fea_pos'] = (0, 2, 3, 4)  # 0,2,3,5 for coref
    # config['WordPos'] = 1
    # config['EmbNames'] = ('Words', 'CAPS', 'POS', 'NER')
    # config['EmbSizes'] = (100, 30, 30, 30)
    # config['const-emb'] = (False, False, False, False)
    # capsVoc = Vocab('res/caps.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    # posVoc = Vocab('res/pos.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    # nerVoc = Vocab('res/ner.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    # config['Vocabs'] = (config['WordId'], capsVoc, posVoc, nerVoc)

    # config['EmbFiles'] = ('res/embedding.txt', None, None, None)

    # config['use_char_conv'] = True
    # config['use_gaz'] = True
    # config['Gazetteers'] = ('PER', 'LOC', 'ORG', 'VEH', 'FAC', 'WEA')
    # config['GazetteerDir'] = 'res/gazetteers'
    # config['gaz_emb_dim'] = 30

    return config

def get_eng_conf():
    config = get_conf_base()
    config['USE_CUDA'] = True
    config['cuda_num'] = 0
    config['lang'] = 'eng'
    config['decoder_output_size'] = 24

    config['OutTags'] = Vocab('res/ner_xx', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['BioOutTags'] = Vocab('res_spa/ner_bio_nottl.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['WordId'] = Vocab('res/voc.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['CharVoc'] = Vocab('res/char.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['max_char'] = 25
    config['char_emb_dim'] = 50
    config['fea_pos'] = (0, 2, 3, 4)  # 0,2,3,5 for coref
    config['WordPos'] = 1
    config['EmbNames'] = ('Words', 'CAPS', 'POS', 'NER')
    config['EmbSizes'] = (100, 30, 30, 30)
    config['const-emb'] = (False, False, False, False)
    capsVoc = Vocab('res/caps.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    posVoc = Vocab('res/pos.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    nerVoc = Vocab('res/ner.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['Vocabs'] = (config['WordId'], capsVoc, posVoc, nerVoc)
    config['EmbFiles'] = ('res/embedding.txt', None, None, None)
    config['use_char_conv'] = True
    config['use_gaz'] = True
    config['Gazetteers'] = ('PER', 'LOC', 'ORG', 'VEH', 'FAC', 'WEA')
    config['GazetteerDir'] = 'res/gazetteers'
    config['gaz_emb_dim'] = 30

    config['model_dir'] = 'eng_bio_model'
    config['train_data'] = 'data/bio_eng_train.txt'
    config['dev_data'] = 'data/bio_eng_dev.txt'
    config['embedding_file'] = 'res/embedding.txt'
    config['eval_emb'] = 'res/emb.txt'

    return config

def get_spa_conf():
    config = get_conf_base()
    config['USE_CUDA'] = True
    config['cuda_num'] = 0
    config['lang'] = 'spa'
    config['decoder_output_size'] = 24

    config['OutTags'] = Vocab('res_spa/ner_xx', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['BioOutTags'] = Vocab('res_spa/ner_bio_nottl.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['WordId'] = Vocab('res_spa/spa_voc.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['CharVoc'] = Vocab('res_spa/char.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['max_char'] = 25
    config['char_emb_dim'] = 50
    config['fea_pos'] = (0, 2, 3, 4)  # 0,2,3,5 for coref
    config['WordPos'] = 1
    config['EmbNames'] = ('Words', 'CAPS', 'POS', 'NER')
    config['EmbSizes'] = (100, 30, 30, 30)
    config['const-emb'] = (False, False, False, False)
    capsVoc = Vocab('res_spa/caps.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    posVoc = Vocab('res_spa/pos.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    nerVoc = Vocab('res_spa/ner.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['Vocabs'] = (config['WordId'], capsVoc, posVoc, nerVoc)
    config['EmbFiles'] = ('res_spa/spa_embedding.txt', None, None, None)
    config['use_char_conv'] = True
    config['use_gaz'] = True
    config['Gazetteers'] = ('PER', 'LOC', 'ORG')
    config['GazetteerDir'] = 'res_spa/gazetteers'
    config['gaz_emb_dim'] = 30

    config['model_dir'] = 'spa_bio_model'
    config['train_data'] = 'data/bio_spa_train.txt'
    config['dev_data'] = 'data/bio_spa_dev.txt'
    config['embedding_file'] = 'res_spa/spa_embedding.txt'
    config['eval_emb'] = 'res_spa/emb.txt'

    return config

def get_cmn_conf():
    config = get_conf_base()
    config['USE_CUDA'] = False
    config['cuda_num'] = 0
    config['lang'] = 'cmn'
    config['decoder_output_size'] = 24

    config['beam_size'] = 8
    config['filter_size'] = 3
    config['OutTags'] = Vocab('res_cmn/ner_xx', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['BioOutTags'] = Vocab('res_cmn/ner_bio_nottl.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['WordId'] = Vocab('res_cmn/cmn_voc_word.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['CharVoc'] = Vocab('res_cmn/cmn_voc_char.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['max_char'] = 25
    config['char_emb_dim'] = 50
    config['fea_pos'] = (0, 2, 3, 4)  # 0,2,3,5 for coref
    config['WordPos'] = 1
    config['EmbNames'] = ('Chars', 'CAPS', 'POS', 'NER')
    config['EmbSizes'] = (200, 30, 30, 30)
    config['const-emb'] = (False, False, False, False)
    capsVoc = Vocab('res_cmn/caps.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    posVoc = Vocab('res_cmn/pos.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    nerVoc = Vocab('res_cmn/ner.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['Vocabs'] = (config['CharVoc'], capsVoc, posVoc, nerVoc)
    config['EmbFiles'] = ('res_cmn/embedding_word.txt', None, None, None)
    config['use_char_conv'] = False
    config['use_gaz'] = True
    config['Gazetteers'] = ('PER', 'LOC', 'ORG')
    config['GazetteerDir'] = 'res_cmn/gazetteers'
    config['gaz_emb_dim'] = 30

    config['model_dir'] = 'cmn_bio_model1'
    config['train_data'] = 'data/bio_cmn_train_nottl.txt'
    config['dev_data'] = 'data/bio_cmn_dev_nottl.txt'
    config['embedding_file'] = 'res_cmn/embedding_word.txt'
    config['char_embedding'] = 'res_cmn/embedding_char.txt'
    config['eval_word_emb'] = 'res_cmn/eval_word_emb.txt'
    config['eval_char_emb'] = 'res_cmn/eval_char_emb.txt'

    return config






# def get_conf_ner():
#     config = get_conf_base()
#     return config
#
#
# def get_conf_coref():
#     config = get_conf_base()
#     return config


# import os
#
#
# def get_conf(task, datadir=None, saveto='saved'):
#     config = None
#     if task == 'ner':
#         config = get_conf_ner()
#         # config['saveto'] = saveto
#         # config['train_data'] = os.path.join(datadir, 'train')
#         # config['test_txt'] = os.path.join(datadir, 'dev')
#         return config
#     elif task == 'coref':
#         config = get_conf_coref()
#         config['saveto'] = saveto
#         config['train_data'] = os.path.join(datadir, 'train.txt')
#         config['test_txt'] = os.path.join(datadir, 'dev.txt')
#         return config
#     else:
#         print 'unknown task {}'.format(task)
#         return None

def get_conf(lang):
    if lang == 'eng':
        config = get_eng_conf()
    elif lang == 'spa':
        config = get_spa_conf()
    elif lang == 'cmn':
        config = get_cmn_conf()
    return config


USE_CUDA = True
max_seq_length = 4
max_label_length = 8
batch_size = 2
hidden_size = 2
n_layers = 2
encoder_outputs_dim = 2
output_size = 3
cuda_num = 0
clip_norm = 1
beam_size = 6
EOS_token = 2
PAD_token = 100
X = 1
att_mode = 'general'

config = get_cmn_conf()
logger = Logger('./train_logs')
