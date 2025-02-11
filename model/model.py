import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import pickle

# loading dataframe
dataframe = pd.read_csv("../data/train.csv")

# filling null values
dataframe = dataframe.fillna('')

# Dropping useless columns
dataframe = dataframe.drop(['id', 'title', 'author'], axis=1)

# Stemming the data
port_stem = PorterStemmer()

#downloading stopwords from nltk
nltk.download('stopwords')

def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con = ' '.join(con)
    return con

# Applying stemming to entire dataframe
dataframe['text'] = dataframe['text'].apply(stemming)

X = dataframe['text']
y = dataframe['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

# Loading model to train
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# predicting the model
prediction = model.predict(X_test)
print(prediction)
# Checking score
print(model.score(X_test, y_test))

# Saving model & Vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))