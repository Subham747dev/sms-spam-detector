import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer as tv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import  accuracy_score,precision_score

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()  # Removed the stray ']' that was causing the syntax error

    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

data = pd.read_csv("C:/Users/Subham/Desktop/spam.csv", encoding='latin1')
data.drop(columns =['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
data.rename(columns={'v1':'target','v2':'text'},inplace=True)
encoder = LabelEncoder()
data['target'] = encoder.fit_transform(data['target'])
data.drop_duplicates(keep = 'first')
data['text'] = data['text'].apply(transform_text)
tb = tv(max_features=3000)
x = tb.fit_transform(data['text']).toarray()
y = data['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

mnb =MultinomialNB()
mnb.fit(x_train,y_train)
y_pred = mnb.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(precision_score(y_test,y_pred))

model = mnb
vectorizer = tb

with open('C:/Users/Subham/Desktop/machine learning projects/sms spam detection/model.pkl','wb') as f:
    pickle.dump(model,f)
with open('C:/Users/Subham/Desktop/machine learning projects/sms spam detection/vectorizer.pkl', 'wb') as f:
    pickle.dump(tb, f)