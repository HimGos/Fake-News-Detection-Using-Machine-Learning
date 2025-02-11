import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import pickle
import logging
import os
from datetime import datetime


# Configure logging
def setup_logger():
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_filename = f"model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    # Loading dataframe
    logger.info("Loading dataset from train.csv")
    dataframe = pd.read_csv("../data/train.csv")
    logger.info(f"Dataset loaded successfully with {len(dataframe)} rows")

    # Filling null values
    logger.info("Filling null values in the dataset")
    dataframe = dataframe.fillna('')

    # Dropping useless columns
    logger.info("Dropping unnecessary columns")
    dataframe = dataframe.drop(['id', 'title', 'author'], axis=1)

    # Downloading stopwords from nltk
    logger.info("Downloading NLTK stopwords")
    try:
        nltk.download('stopwords')
    except Exception as e:
        logger.error(f"Failed to download NLTK stopwords: {str(e)}")
        raise

    # Initialize Porter Stemmer
    port_stem = PorterStemmer()


    def stemming(content):
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


    # Applying stemming to entire dataframe
    logger.info("Applying stemming to the dataset")
    dataframe['text'] = dataframe['text'].apply(stemming)

    # Splitting the dataset
    logger.info("Splitting dataset into training and testing sets")
    X = dataframe['text']
    y = dataframe['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF Vectorization
    logger.info("Performing TF-IDF vectorization")
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)

    # Training the model
    logger.info("Training Decision Tree Classifier")
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Model evaluation
    prediction = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    logger.info(f"Model accuracy: {accuracy:.4f}")

    # Saving model & Vectorizer
    logger.info("Saving model and vectorizer")
    try:
        pickle.dump(model, open('model.pkl', 'wb'))
        pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
        logger.info("Model and vectorizer saved successfully")
    except Exception as e:
        logger.error(f"Error saving model or vectorizer: {str(e)}")
        raise

except FileNotFoundError as e:
    logger.error(f"File not found error: {str(e)}")
    raise
except Exception as e:
    logger.error(f"An unexpected error occurred: {str(e)}")
    raise
