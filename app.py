import re
import pickle
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

port_stem = PorterStemmer()
vectorizer = pickle.load(open('model/tfidf.pkl', 'rb'))
model = pickle.load(open('model/model.pkl', 'rb'))

#downloading stopwords from nltk
nltk.download('stopwords')

def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con

def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vectorizer.transform(input_data)
    prediction = model.predict(vector_form1)
    return prediction

if __name__ == '__main__':
    st.title('Fake News Detection App')
    st.subheader("by - Himanshu Goswami")
    sentence = st.text_area("Enter your news content here", "",height=200)
    predict_btt = st.button("Predict")
    if predict_btt:
        prediction_class=fake_news(sentence)
        print(prediction_class)
        if prediction_class == [0]:
            st.success('Reliable News')
        if prediction_class == [1]:
            st.warning('Unreliable News')
