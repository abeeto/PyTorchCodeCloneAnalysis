# https://paul-hyun.github.io/vocab-with-sentencepiece/

import sentencepiece as spm
import os

# 함수의 입력으로는 corpus(말뭉치), prefix(생성할 파일 이름), vocab_size(vocabulary 개수) 등을 받습니다.
# -–input: 망뭉치 파일 경로를 입력합니다.
# -–model_prefix: 생성할 파일 이름을 입력합니다. model_prefix.model, model_prefix.vocab 두 개의 파일이 생성됩니다.
# –vocab_size: 생성할 vocabulary의 개수를 입력합니다. 특수문자 개수 7을 추가해서 지정했습니다.
# –model_type: Vocabulary를 학습할 방법을 입력합니다. 기본값인 ‘unigram’을 입력합니다.
# –max_sentence_length: 한 문장의 최대길이를 의미합니다.
# –pad_id: 짧은 문장의 길이를 맞춰줄 pad token의 일련번호 의미합니다.
# –pad_piece: pad token의 문자열을 의미합니다.
# –unk_id: Vocabulary에 없는 token을 처리할 unk token의 일련번호 의미합니다.
# –unk_piece: unk token의 문자열을 의미합니다.
# –bos_id: 문장의 시작을 의미하는 bos token의 일련번호 의미합니다.
# –bos_piece: bos token의 문자열을 의미합니다.
# –eos_id: 짧은 문장의 길이를 맞춰줄 pad token의 일련번호 의미합니다.
# –eos_piece: pad token의 문자열을 의미합니다.
# –user_defined_symbols: 추가적으로 필요한 token을 정의합니다. 일련번호는 특수 토큰 이후부터 순서대로 부여됩니다.
# –input_sentence_size: 한국어위키의 말뭉치의 경우는 데이터가 커서 vocabulary를 학습하는데 시간이 많이 걸립니다. 시간 단축을 위해 sampling을 해서 학습하도록 sampling 개수를 정의합니다. 전체 말뭉치를 학습하기를 원하면 ‘input_sentence_size’, ‘shuffle_input_sentence’를 제거하면 됩니다.
# –shuffle_input_sentence: 말뭉치의 순서를 섞을지 여부를 결정합니다.


def train_sentencepiece(corpus, prefix, vocab_size=32000):
    """
    sentencepiece를 이용해 vocab 학습
    :param corpus: 학습할 말뭉치
    :param prefix: 저장할 vocab 이름
    :param vocab_size: vocab 개수
    """
    # spm.SentencePieceTrainer.train(
    #     f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" +  # 7은 특수문자 개수
    #     " --model_type=unigram" +
    #     " --max_sentence_length=999999" +  # 문장 최대 길이
    #     " --pad_id=0 --pad_piece=[PAD]" +  # pad token 및 id 지정
    #     " --unk_id=1 --unk_piece=[UNK]" +  # unknown token 및 id 지정
    #     " --bos_id=2 --bos_piece=[BOS]" +  # begin of sequence token 및 id 지정
    #     " --eos_id=3 --eos_piece=[EOS]" +  # end of sequence token 및 id 지정
    #     " --user_defined_symbols=[SEP],[CLS],[MASK]" +  # 기타 추가 토큰 SEP: 4, CLS: 5, MASK: 6
    #     " --input_sentence_size=100000" +  # 말뭉치에서 셈플링해서 학습
    #     " --shuffle_input_sentence=true")  # 셈플링한 말뭉치 shuffle
    spm.SentencePieceTrainer.train(
        f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" +  # 7은 특수문자 개수
        " --model_type=unigram" +
        " --max_sentence_length=999999" +  # 문장 최대 길이
        " --pad_id=0 --pad_piece=[PAD]" +  # pad token 및 id 지정
        " --unk_id=1 --unk_piece=[UNK]" +  # unknown token 및 id 지정
        " --bos_id=2 --bos_piece=[BOS]" +  # begin of sequence token 및 id 지정
        " --eos_id=3 --eos_piece=[EOS]" +  # end of sequence token 및 id 지정
        " --user_defined_symbols=[SEP],[CLS],[MASK]")  # 셈플링한 말뭉치 shuffle


if __name__ == '__main__':

    if False:
        BASE = 'D:/Datas/NLP/번역/ai허브/한국어_영어_번역말뭉치'

        # han_path_tmp = '뉴스_한국어.txt'
        # eng_path_tmp = '뉴스_영어.txt'

        # han_path = os.path.join(BASE, han_path_tmp)
        # eng_path = os.path.join(BASE, eng_path_tmp)

        han_path = 'newsKor.txt'
        eng_path = 'newsEng.txt'

        # vocab 학습
        # train_sentencepiece(han_path, "news_han_32000")
        # train_sentencepiece(eng_path, "news_eng_32000")
        train_sentencepiece(han_path, "news_han_28000", 28000)
        train_sentencepiece(eng_path, "news_eng_18000", 18000)

    if True:
        # 학습된 결과 확인
        vocab_han = spm.SentencePieceProcessor()
        vocab_eng = spm.SentencePieceProcessor()

        vocab_han.load("news_han_28000.model")
        vocab_eng.load("news_eng_18000.model")

        text_han = '안녕하세요. 저는 윤희동입니다.'
        token_han = vocab_han.encode_as_pieces(text_han)
        token_eng = vocab_eng.encode_as_pieces('hello. i am yoonheedong.')

        id_han = vocab_han.encode(text_han)
        print(f'입력된 텍스트: {text_han}')
        print(f'토큰화: {token_han}')
        print(f'id   : {id_han}')


        print(vocab_han.decode([339, 7575, 519, 237, 3851, 8, 703, 10, 1472, 878, 199, 1166, 8]))
        bb =0


        # print(token_eng)