import nltk

# ✅ Clear any previous paths and set only the clean one
nltk.data.path.clear()
nltk.data.path.append("/tmp/nltk_data")

# 🔽 Import tokenizer after setting clean path
from nltk.tokenize import word_tokenize

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer as tv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english')]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Only run training when executed directly
if __name__ == "__main__":
    data = pd.read_csv("spam.csv", encoding='latin1')
    data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    data.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

    encoder = LabelEncoder()
    data['target'] = encoder.fit_transform(data['target'])

    data.drop_duplicates(keep='first', inplace=True)
    data['text'] = data['text'].apply(transform_text)

    tb = tv(max_features=3000)
    x = tb.fit_transform(data['text']).toarray()
    y = data['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    mnb = MultinomialNB()
    mnb.fit(x_train, y_train)

    with open('model.pkl', 'wb') as f:
        pickle.dump(mnb, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(tb, f)

    print("Model trained and saved successfully.")
