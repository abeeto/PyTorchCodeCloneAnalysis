import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


data = pd.read_csv('spam.csv')
data.dropna(how="any")
data['Category'] = data['Category'].replace('ham',0)
data['Category'] = data['Category'].replace('spam',1)

print(data)
print('____________________________________')

# токенизация текста
data_with_tokens = data.copy()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
for i in range(len(data['Message'])):
    data_with_tokens['Message'][i] = nltk.word_tokenize(data['Message'][i])

print(data_with_tokens)

print('____________________________________')

# Нормализация текста
data_with_norm = data.copy()

# В нижний ргистр
data_with_norm['Message'] = data_with_norm['Message'].str.lower()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

for i in range(len(data_with_norm['Message'])):
    # Удаляем числа
    data_with_norm['Message'][i] = re.sub(r'\d+','',data_with_norm['Message'][i])
    # Удаляем знаки препинания
    data_with_norm['Message'][i] = re.sub(r'[^\w\s]','', data_with_norm['Message'][i])
    # Удаляем путоту
    data_with_norm['Message'][i] = data_with_norm['Message'][i].strip()
    #Удаляем стоп слова
    data_with_norm['Message'][i] = data_with_norm['Message'][i].split()
    no_stpwords_string=""
    for j in data_with_norm['Message'][i]:
        if not j in stop_words:
            no_stpwords_string += j+' '
    
    data_with_norm['Message'][i] = no_stpwords_string

    # Леммантизация слов
    data_with_norm['Message'][i] = lemmatizer.lemmatize(data_with_norm['Message'][i])
    

print(data_with_norm)

