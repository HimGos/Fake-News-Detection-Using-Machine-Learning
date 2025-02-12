import re
import pickle
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import logging
import os
from datetime import datetime

# Configure logging
def setup_logger():
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_filename = f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_directory, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

try:
    # Initialize Porter Stemmer
    port_stem = PorterStemmer()

    # Load models
    logger.info("Loading trained model and vectorizer")
    vectorizer = pickle.load(open('model/tfidf.pkl', 'rb'))
    model = pickle.load(open('model/model.pkl', 'rb'))
    logger.info("Model and vectorizer loaded successfully")
    
    # Download NLTK stopwords
    nltk.download('stopwords')


    def stemming(content):
        """
        Performs preprocessing and stemming on the input text content.
    
        Args:
            content (str): The input text to preprocess and stem.
    
        Returns:
            str: The processed and stemmed text.
    
        Raises:
            Exception: If any error occurs during the stemming process.
        """
        try:
            con = re.sub('[^a-zA-Z]', ' ', content)
            con = con.lower()
            con = con.split()
            con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
            con = ' '.join(con)
            return con
        except Exception as e:
            logger.error(f"Error in stemming process: {str(e)}")
            raise

    def fake_news(news):
         """
        Predicts whether the input news is fake or reliable.

        Args:
            news (str): Input news text content

        Returns:
            List[int]: Prediction result where:
                - [0] indicates reliable news
                - [1] indicates unreliable news

        Raises:
            ValueError: If news content is empty or invalid
            ModelError: If prediction fails
        """
        try:
            if not news.strip():
                logger.warning("Empty news content received")
                raise ValueError("News content cannot be empty")

            logger.info("Processing news content")
            news = stemming(news)
            input_data = [news]
            vector_form1 = vectorizer.transform(input_data)
            prediction = model.predict(vector_form1)
            logger.info(f"Prediction complete. Result: {prediction[0]}")
            return prediction
        except Exception as e:
            logger.error(f"Error in fake news detection: {str(e)}")
            raise

    if __name__ == '__main__':
        st.title('Fake News Detection App')
        st.subheader("by - Himanshu Goswami")

        sentence = st.text_area("Enter your news content here", "", height=200)
        predict_btt = st.button("Predict")

        if predict_btt:
            try:
                if not sentence.strip():
                    st.error("Please enter some news content before prediction")
                    logger.warning("Prediction attempted with empty content")
                else:
                    logger.info("Processing prediction request")
                    prediction_class = fake_news(sentence)

                    if prediction_class == [0]:
                        logger.info("Prediction: Reliable News")
                        st.success('Reliable News')
                    if prediction_class == [1]:
                        logger.info("Prediction: Unreliable News")
                        st.warning('Unreliable News')

            except Exception as e:
                error_message = f"An error occurred during prediction: {str(e)}"
                logger.error(error_message)
                st.error(error_message)

except Exception as e:
    logger.error(f"Application startup error: {str(e)}")
    st.error("An error occurred while starting the application. Please check the logs.")
