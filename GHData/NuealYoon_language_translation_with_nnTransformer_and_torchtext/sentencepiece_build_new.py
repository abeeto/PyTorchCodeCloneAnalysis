# 스페셜토큰이 나오지 않길래 새로운 코드 찾음
# https://keep-steady.tistory.com/37

#############################
# 2. 구글의 SentencePiece 학습
#############################

# %%time
import os
import sentencepiece as spm

# spm_train --input=data/train_tokenizer.txt  --model_prefix=sentencepiece/sp --vocab_size=32000 character_coverage=1.0 --model_type="unigram"

# input_file = 'data/train_tokenizer.txt'
# input_file = 'newsKor.txt'
# vocab_size = 28000
# sp_model_root='sentencepiece_kor'
input_file = './data/newsKor.txt'
vocab_size = 18000
sp_model_root='sentencepiece_eng'
if not os.path.isdir(sp_model_root):
    os.mkdir(sp_model_root)
sp_model_name = 'tokenizer_%d' % (vocab_size)
sp_model_path = os.path.join(sp_model_root, sp_model_name)
model_type = 'unigram'  # 학습할 모델 선택, unigram이 더 성능이 좋음'bpe'
character_coverage  = 1.0  # 전체를 cover 하기 위해, default=0.9995
user_defined_symbols = '[PAD],[UNK],[CLS],[SEP],[MASK],[BOS],[EOS],[UNK0],[UNK1],[UNK2],[UNK3],[UNK4],[UNK5],[UNK6],[UNK7],[UNK8],[UNK9],[unused0],[unused1],[unused2],[unused3],[unused4],[unused5],[unused6],[unused7],[unused8],[unused9],[unused10],[unused11],[unused12],[unused13],[unused14],[unused15],[unused16],[unused17],[unused18],[unused19],[unused20],[unused21],[unused22],[unused23],[unused24],[unused25],[unused26],[unused27],[unused28],[unused29],[unused30],[unused31],[unused32],[unused33],[unused34],[unused35],[unused36],[unused37],[unused38],[unused39],[unused40],[unused41],[unused42],[unused43],[unused44],[unused45],[unused46],[unused47],[unused48],[unused49],[unused50],[unused51],[unused52],[unused53],[unused54],[unused55],[unused56],[unused57],[unused58],[unused59],[unused60],[unused61],[unused62],[unused63],[unused64],[unused65],[unused66],[unused67],[unused68],[unused69],[unused70],[unused71],[unused72],[unused73],[unused74],[unused75],[unused76],[unused77],[unused78],[unused79],[unused80],[unused81],[unused82],[unused83],[unused84],[unused85],[unused86],[unused87],[unused88],[unused89],[unused90],[unused91],[unused92],[unused93],[unused94],[unused95],[unused96],[unused97],[unused98],[unused99]'

input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --user_defined_symbols=%s --model_type=%s --character_coverage=%s'
cmd = input_argument%(input_file, sp_model_path, vocab_size,user_defined_symbols, model_type, character_coverage)

spm.SentencePieceTrainer.Train(cmd)
print('train done')

##################################################
# 3. 학습한 BPE 모델을 load하여 subword tokenize 수행
##################################################

## check
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load('{}.model'.format(sp_model_path))

tokens = sp.encode_as_pieces('나는 오늘 아침밥을 먹었다.')
ids = sp.encode_as_ids('나는 오늘 아침밥을 먹었다.')

print(ids)
print(tokens)

tokens = sp.decode_pieces(tokens)
ids = sp.decode_ids(ids)

print(ids)
print(tokens)

tokens = sp.encode(['[BOS]'])

print(tokens)
print(tokens)