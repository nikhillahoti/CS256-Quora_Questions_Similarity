import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
df_train = pd.read_csv('/Users/vedashreebhandare/Documents/CS256/DupQuestiondata/train.csv')
print(df_train.head())
print('Type of column quest2',df_train['question2'].dtype)

exit(0)

print('Hello')

def remove_stop_words():
    q1words = []
    q2words = []
    stops = set(stopwords.words("english"))
    for question in df_train['question1']:
        for word in str(question).lower().split():
                if word not in stops:
                    q1words.append(word)
                    #q1words[word] = 1
    for question in df_train['question2']:
        for word in str(question).lower().split():
                if word not in stops:
                    q2words.append(word)
                    #q2words[word] = 1 

    print('Length of q1',len(q1words))
    if len(q1words)> 0:
        # for i in range(0, 100):
        #     print(q1words[i])     
        print('Length of q2',len(q2words))
    
    
    
    wnl = WordNetLemmatizer()
    for idx, item in enumerate(q1words):
        q1words[idx] = wnl.lemmatize(item)
    for idx, item in enumerate(q2words):
        q2words[idx] = wnl.lemmatize(item)
    # for i in range(0, 100):
    #     print(q1words[i])
        
        
    vectorizer = TfidfVectorizer()
    q1words.extend(q2words)
    # print(len(q1words))
    # print("Here ")
    # exit(0)
    combinedWordSet = set(q1words)
    vectorizer.fit(combinedWordSet)
    q1VectorList =[]
    q2VectorList =[]
    for question in df_train['question1']:
        q1VectorList.append(vectorizer.transform(question.split()))
    for question in df_train['question2']:
        q2VectorList.append(vectorizer.transform(str(question).split()))
    print('Q1Vector Shape-->',len(q1VectorList))
    print('Q2Vector Shape-->',len(q2VectorList))
    
    # q1Vector = vectorizer.fit_transform(q1words)
    # print(q1Vector.shape)
    # q2Vector = vectorizer.fit_transform(q2words)
    # print(q2Vector.shape)
        

        

remove_stop_words()      
