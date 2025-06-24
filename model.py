import nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer as tv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')

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
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

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

    # Save model and vectorizer
    with open('model.pkl', 'wb') as f:
        pickle.dump(mnb, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(tb, f)

    # Optional: print accuracy
    print("Model trained and saved successfully.")
