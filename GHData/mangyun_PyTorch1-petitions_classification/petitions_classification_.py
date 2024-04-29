

#크롤링
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup 
import time

result = pd.DataFrame()  # 빈 데이터프레임 생성                                  

for i in range(584274, 595226):# 청원의 수 584274 ~ 595225+1까지의 범위 지정
    URL = "http://www1.president.go.kr/petitions/"+str(i)# 청원불러오기
 
    response = requests.get(URL)    # url로 페이지에 접속해서 response에 저장
    html = response.text    # 페이지에서 F12를 누르면 나오는 HTML코드 저장                                  
    soup = BeautifulSoup(html, 'html.parser')    #HTML의 코드를 추출하는 모듈 BeautifulSoup이용. html.parser로 파이썬이 이해하도록 객체 변환

    title = soup.find('h3', class_='petitionsView_title') #HTML에 해당하는 태그와 속성을 이용해서 제목을 가져옴
    count = soup.find('span', class_='counter')          # 위와 같이 참여인원을 가져옴 

    for content in soup.select('div.petitionsView_write > div.View_write'):# 속성값하위이므로 >를 이용해 청원내용을 저장
        content                                         

    a=[] #청원정보를 저장할 배열생성
    for tag in soup.select('ul.petitionsView_info_list > li'): # 청원정보중 두번째 내용만 필요하므로 인덱스 0다음 1지정
        a.append(tag.contents[1])




    #데이터프레임 또 생성
    if len(a) != 0:
        df1=pd.DataFrame({ 'start' : [a[1]],               #청원시작날짜 지정
                           'end' : [a[2]],                 #청원 마지막 날짜 지정
                           'category' :  [a[0]],            #카테고리 저장
                           'count' : [count.text],            #참여인원저장
                           'title': [title.text],              #제목저장
                           'content': [content.text.strip()[0:13000]]   # strip()을 추가해 공백을 제거한뒤, 엑셀저장을 위해 13,000자로 제한. 엑셀은 한셀의 최대글자가 32,767개이기때문에 토그나이징과 제목과 결합했을때 초과하지않기 위함.                           
                         })

        result=pd.concat([result, df1])                 # 위에서 하나의 청원을 하나의 데이터프레임으로 생성했으니, 여러 글들을 여러 데이터프레임으로 누적 병합       
        result.index = np.arange(len(result))             #0 부터 프레임의 개수-1 까지의 값을 갖는 배열 인덱스 생성

    #국민청원페이지는 연속접근할 경우 접근이 불가능함. 따라서 60건을 크롤링할때마다 90초를 멈춘뒤 재작업. 이처럼 웹페이지마다 크롤링이 불가한 경우도있고, 코드로 예외처리를 해야함.
    if i % 60 == 0:          # 90초간 시스템 멈춤을 알림과 동시에 청원의 순번, 현재 시각, 현재까지 크롤링한 데이터의 수를 출력.                          
        print("Sleep 90seconds. Count:" + str(i)           
              +",  Local Time:"+ time.strftime('%Y-%m-%d', time.localtime(time.time()))
              +" "+ time.strftime('%X', time.localtime(time.time()))
              +",  Data Length:"+ str(len(result)))        
        time.sleep(90) 




#크롤링 데이터 확인
#print(result.shape)# (10881, 6)

df = result () # 원본데이터를 유지하기 위해 df로 복사

# 데이터 엑셀로 저장
df.to_csv('C:/Users/auddb/OneDrive/바탕 화면/분류/crawling.csv', index = False, encoding = 'utf-8-sig') # 데이터 엑셀로 저장





#데이터 전처리
import re #정규 표현식 지원 모듈, 파이썬만이 아니라, 타 프로그래밍 언어에서도 사용

def remove_white_space(text):# 공백문자제거 함수, \t탭 \r\n엔터 \f줄바꿈 \v수직탭 문자를 공백으로 치환
    text = re.sub(r'[\t\r\n\f\v]', ' ', str(text)) 
    return text

def remove_special_char(text):# 특수문자 제거함수, ㄱ~ㅣ까지의 자음, 가~힣까지의 한글, 0~9까지의 숫자에 해당하지 않으면 공백으로 치환
    text = re.sub('[^ ㄱ-ㅣ가-힣 0-9]+', ' ', str(text))
    return text

df.title = df.title.apply(remove_white_space)# 제목에 공백제거
df.title = df.title.apply(remove_special_char)# 제목에 특수문자 제거

df.content = df.content.apply(remove_white_space)# 내용에 공백제거
df.content = df.content.apply(remove_special_char)# 내용에 특수문자 제거





#토크나이징 및 변수 생성
#토크나이징
from konlpy.tag import Okt #konlpy는 형태소 분석기 패키지, konply.tag에 한국어 지원 여러 클래스 중 Okt 선정

okt = Okt()

df['title_token'] = df.title.apply(okt.morphs)# 제목을 형태소 단위로 토크나이징
df['content_token'] = df.content.apply(okt.nouns)# 내용을 명사 단위로 토크나이징

#파생변수 생성
df['token_final'] = df.title_token + df.content_token #형태소인 제목과 명사인 내용을 더해 token_final에 저장

df['count'] = df['count'].replace({',' : ''}, regex = True).apply(lambda x : int(x))#참여인원은 천단위마다 ,이 있어서 object로 인식하기때문에, 제거한뒤 int로 변환

df['label'] = df['count'].apply(lambda x: 'Yes' if x>=1000 else 'No') #1000명 이상이면 label에 yes에 저장, 미만이면 no 저장

df_drop = df[['token_final', 'label']] # 분석에 필요한 token_final과 label만 저장

#데이터 엑셀로 저장
df_drop.to_csv('C:/Users/auddb/OneDrive/바탕 화면/분류/df_drop.csv', index = False, encoding = 'utf-8-sig')





#단어 임베딩
#단어 임베딩
from gensim.models import Word2Vec # gensim.models는 자연어 처리모델을 지원, 그 중 Word2Vec 사용

embedding_model = Word2Vec(df_drop['token_final'], # 임베딩 벡터를 생성할 대상 데이터
                           sg = 1, # 1 : skip-gram, 0 : CBOW
                           size = 100, # 벡터의 크기
                           window = 2, # 문맥파악을 위해 앞,뒤 토큰 수
                           min_count = 1, #전체토큰에서 일정 횟수 이상 등장하지않는다면 제외
                           workers = 4 # 실행할 병렬 프로세스, 보통 4~6 사용
                           )

#print(embedding_model) 총 43,937개의 토큰 저장

#model_result = embedding_model.wv.most_similar("음주운전") 모델에서 음주운전과 유사한 벡터값을 저장
#print(model_result) 출력


#임베딩 모델 저장 및 로드
from gensim.models import KeyedVectors #임베딩 모델을 부르는 클래스

embedding_model.wv.save_word2vec_format('C:/Users/auddb/OneDrive/바탕 화면/분류/petitions_tokens_w2v') # 로컬데이터에 모델 저장
loaded_model = KeyedVectors.load_word2vec_format('C:/Users/auddb/OneDrive/바탕 화면/분류/petitions_tokens_w2v') # 모델 로드

#model_result = loaded_model.most_similar("음주운전")
#print(model_result) 이상없이 로드 되었는지 확인






#실험 설계
#데이터셋 분할 및 저장
from numpy.random import RandomState #데이터 모델 

rng = RandomState() #상태지정

tr = df_drop.sample(frac=0.8, random_state=rng) #전체 데이터의 80%를 train에 저장
val = df_drop.loc[~df_drop.index.isin(tr.index)] # 나머지를 validation에 저장, 원래는 test까지도 나누지만 지금은 데이터가 적기때문에 test를 생략

tr.to_csv('C:/Users/auddb/OneDrive/바탕 화면/분류/train.csv', index=False, encoding='utf-8-sig') # 분할결과를 data폴더에 csv 형식으로 저장
val.to_csv('C:/Users/auddb/OneDrive/바탕 화면/분류/validation.csv', index=False, encoding='utf-8-sig') # 분할 결과를 data폴더에 csv형식으로 저장


#field클래스 정의
import torchtext #자연어 처리를 위한 라이브러리
from torchtext.data import Field #field 클래스는 토크나이징 및 단어장 생성

def tokenizer(text): # 토크나이징 정의 함수
    text = re.sub('[\[\]\']', '', str(text)) #token_final에서 문자 [, ], '를 제거해 형태변경, 안하면 전체를 하나의 토큰으로 인식
    text = text.split(', ')# 다시 구분자로 구분
    return text # 반환

TEXT = Field(tokenize=tokenizer) # 여러옵션 중, tokenize 옵션만 정의
LABEL = Field(sequential = False) # 여러옵션 중,  label 이라 yes, no이기 때문에 순서가 불필요


#데이터 불러오기
from torchtext.data import TabularDataset # 데이터를 읽어 데이터셋의 생성 지원

train, validation = TabularDataset.splits( # 데이터 셋 생성
    path = 'C:/Users/auddb/OneDrive/바탕 화면/분류/',# 데이터 경로
    train = 'train.csv',#훈련 지정
    validation = 'validation.csv',#검증지정
    format = 'csv',#csv, tsv, json 포맷 지정
    fields = [('text', TEXT), ('label', LABEL)], # 앞에서 했던 field 설정
    skip_header = True # 데이터 첫 행이 컬럼명인 경우, True로 지정
)

#print("Train:", train[0].text,  train[0].label) 출력
#print("Validation:", validation[0].text, validation[0].label) 확인


#단어장 및 dataLoader 정의
import torch # 파이토치 기본 모듈
from torchtext.vocab import Vectors #임베딩 벡터 생성을 위한 클래스
from torchtext.data import BucketIterator#bucketiterator는 데이터셋에서 배치사이즈만큼 데이터 로드하는 iterator를 지원

vectors = Vectors(name="C:/Users/auddb/OneDrive/바탕 화면/분류/petitions_tokens_w2v") # 사전에 훈련된 벡터 지정

TEXT.build_vocab(train, vectors = vectors, min_freq = 1, max_size = None)# train 데이터의 단어장을 생성, vectors는 단어장에 세팅하는 옵션, min_freq는 등장빈도가 일정 수준 이상인 토크만 지원, 1회라도 등장한 토큰은 포함
 #max_size는 전체 vocabsize에 제한을 둠. 전체 토큰을 활용해야 none으로 지정
LABEL.build_vocab(train)

vocab = TEXT.vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# gpu가 가능하다면 gpu, 아니면 cpu 사용

train_iter, validation_iter = BucketIterator.splits(#splits함수를 이용해, 배치사이즈만큼 로드
    datasets = (train, validation),#로드한 데이터셋
    batch_size = 8,#배치데이터 크기
    device = device,#장비할당
    sort = False#전체 데이터 정렬 여부
)

#print('임베딩 벡터의 개수와 차원 : {} '.format(TEXT.vocab.vectors.shape))




#TextCNN
#TextCNN 모델링
import torch.nn as nn   # 신경망을 쌓기위한 모듈
import torch.optim as optim  # 최적화 함수를 위한 모듈
import torch.nn.functional as F #활성함수, 손실함수 등 모든 함수가 포함된 모듈

class TextCNN(nn.Module): #기본함수를 포함하는 nn.Module을 상속받는 클래스 생성
    
    def __init__(self, vocab_built, emb_dim, dim_channel, kernel_wins, num_class):
        
        super(TextCNN, self).__init__()# 상속받아 이용
        
        self.embed = nn.Embedding(len(vocab_built), emb_dim)# 임베딩 설정을 위해 vocab_size*embedding dimension(40348*100) 크기를 만듬
        self.embed.weight.data.copy_(vocab_built.vectors)      # 이전 학슴한 벡터값들을 가져옴
    
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])#임베딩 결과를 filter를 생성
        #1 : input 채널
        #dim_channel: output채널, 크기와 같음 10.
        #w : filter의 크기, kernel_wins가 [3,4,5] w도 3,4,5를 가짐
        #emb_dim : 임베딩 벡터의 크기 100
        self.relu = nn.ReLU() # activation map을 생성하는 relu 함수 정의                
        self.dropout = nn.Dropout(0.4)         # 오버피팅을 방지하기 위해 랜덤하게 가중치의 40%에 0을 넣어 계산
        self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_class)     # 클래스의 fully connected layer 생성 (3*10, 2)
        
    def forward(self, x):  # 입력했을때부터 output 계산까지의 과정
      
        emb_x = self.embed(x)  #임베딩 정보 전달         
        emb_x = emb_x.unsqueeze(1)  #두번째 인자에 1을 추가하여 차원 증가

        con_x = [self.relu(conv(emb_x)) for conv in self.convs]       #리스트에 저장된 filter 3,4,5를 통해 feature map 3개를 리스트형태로 저장 후, relu를 통과해 activation map 생성
    

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]  #max pooling을 진행해 리스트 형태로 이루어진 pooling layer([dim_channel(10)*1],[dim_channel(10)*1],[dim_channel(10)*1])  
        
        fc_x = torch.cat(pool_x, dim=1) # 풀링한 1차원벡터를 concat 하여 하나의 fc layer(pool layer크기10*3)생성
        fc_x = fc_x.squeeze(-1)       #차원을 줄여 (30*1)형태인 최종 fc layer 생성
        fc_x = self.dropout(fc_x)         # dropout 적용

        logit = self.fc(fc_x)     # (30*1)크기의 fc_x를 fc 함수에 통과시켜 (1*2) 크기인 fc layer의 propagation logit 값 계산
        
        return logit # 반환


#모델 학습 함수 정의
def train(model, device, train_itr, optimizer): # train 함수 정의
    
    model.train()                       # TextCNN 모델을 Train mode 로 변경하여 업데이트 하게 함. 위의 train이랑은 다름
    corrects, train_loss = 0.0,0        # 올바른 클래스로 분류한 기준에 대한 변수 지정
    
    for batch in train_itr:
        
        text, target = batch.text, batch.label      #미니배치로 저장된 텍스트와 레이블 지정
        text = torch.transpose(text, 0, 1)          #연산을 위해 텍스트를 역행렬로 변환
        target.data.sub_(1)                                 #레이블인 target 각 값을 1씩 줄임
        text, target = text.to(device), target.to(device)  # 장비 할당

        optimizer.zero_grad()                           # 최적화를 위해 손실 초기화
        logit = model(text)                         # 텍스트로 input을 이용해 output 계산
    
        loss = F.cross_entropy(logit, target)   #label 비교
        loss.backward()  # 역전파 계산
        optimizer.step()  #parameter 값을 업데이트
        
        train_loss += loss.item()    #미니배치에서의 loss 값을 train_loss 값에 누적 더함
        result = torch.max(logit,1)[1] #인덱스별로 계산된 값에서 더 큰 확률을 가진 클래스 저장
        corrects += (result.view(target.size()).data == target.data).sum() # 예측값과 레이블이 맞으면 횟수 증가
        
    train_loss /= len(train_itr.dataset) # 현재 손실값에 미니배치개수만큼 나눠 평균 손실 값 계산
    accuracy = 100.0 * corrects / len(train_itr.dataset) # 정확도 계산

    return train_loss, accuracy # 반환



#모델 평가 함수, 위와 똑같음
def evaluate(model, device, itr):
    
    model.eval()
    corrects, test_loss = 0.0, 0

    for batch in itr:
        
        text = batch.text
        target = batch.label
        text = torch.transpose(text, 0, 1)
        target.data.sub_(1)
        text, target = text.to(device), target.to(device)
        
        logit = model(text)
        loss = F.cross_entropy(logit, target)

        test_loss += loss.item()
        result = torch.max(logit,1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()

    test_loss /= len(itr.dataset) 
    accuracy = 100.0 * corrects / len(itr.dataset)
    
    return test_loss, accuracy


#모델 학습 및 성능 확인
model = TextCNN(vocab, 100, 10, [3, 4, 5], 2).to(device)
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer = optim.Adam(model.parameters(), lr=0.001) # 역전파를 통해 업데이트할때 Adam 이용

best_test_acc = -1

for epoch in range(1, 3+1):
 
    tr_loss, tr_acc = train(model, device, train_iter, optimizer) #학습 함수 실행
    print('Train Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, tr_loss, tr_acc))
    
    val_loss, val_acc = evaluate(model, device, validation_iter)# 평가 함수 실행
    print('Valid Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, val_loss, val_acc))
        
    if val_acc > best_test_acc:
        best_test_acc = val_acc
        
        print("model saves at {} accuracy".format(best_test_acc))
        torch.save(model.state_dict(), "TextCNN_Best_Validation")
    
    print('-----------------------------------------------------------------------------')


