import unittest
import pandas as pd
import numpy as np
import pickle
import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from app import stemming, fake_news, setup_logger
from model.model import port_stem

class TestTextPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        nltk.download('stopwords')
        cls.logger = setup_logger()

    def test_stemming_basic(self):
        """Test basic stemming functionality"""
        test_text = "Running tests for our application"
        result = stemming(test_text)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        self.assertNotIn("Running", result)  # Should be stemmed
        self.assertIn("run", result.split())

    def test_stemming_special_characters(self):
        """Test stemming with special characters"""
        test_text = "Hello! This is a test... With special-characters & numbers 123"
        result = stemming(test_text)
        self.assertNotIn("!", result)
        self.assertNotIn("...", result)
        self.assertNotIn("123", result)

    def test_stemming_empty_input(self):
        """Test stemming with empty input"""
        with self.assertRaises(ValueError):
            fake_news("")

    def test_stemming_stopwords(self):
        """Test if stopwords are removed"""
        test_text = "This is a test of the stopwords removal"
        result = stemming(test_text)
        stop_words = set(stopwords.words('english'))
        result_words = set(result.split())
        # Check if any stopwords remain
        self.assertEqual(len(result_words.intersection(stop_words)), 0)

class TestModelPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load model and vectorizer for testing"""
        try:
            cls.vectorizer = pickle.load(open('model/tfidf.pkl', 'rb'))
            cls.model = pickle.load(open('model/model.pkl', 'rb'))
        except FileNotFoundError:
            raise unittest.SkipTest("Model files not found. Skipping model tests.")

    def test_prediction_fake_news(self):
        """Test prediction for likely fake news"""
        fake_news_text = """BREAKING: Aliens have officially made contact with Earth! 
        Government sources confirm secret meetings with extraterrestrial beings. 
        World leaders preparing to announce global changes!"""
        prediction = fake_news(fake_news_text)
        self.assertIn(prediction[0], [0, 1])  # Should return either 0 or 1

    def test_prediction_real_news(self):
        """Test prediction for likely real news"""
        real_news_text = """The city council met yesterday to discuss the new budget proposal. 
        The meeting was attended by council members and local citizens. 
        Several key infrastructure projects were approved."""
        prediction = fake_news(real_news_text)
        self.assertIn(prediction[0], [0, 1])

    def test_prediction_input_length(self):
        """Test predictions with different input lengths"""
        short_text = "Very short text"
        with self.assertRaises(ValueError):
            fake_news(short_text)

class TestModelTraining(unittest.TestCase):
    def test_vectorizer_functionality(self):
        """Test TF-IDF vectorizer functionality"""
        test_texts = [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third document."
        ]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(test_texts)
        self.assertTrue(X.shape[0] == 3)  # Should have 3 documents
        self.assertTrue(X.shape[1] > 0)   # Should have features

    def test_model_accuracy(self):
        """Test if model meets minimum accuracy threshold"""
        if hasattr(self, 'model'):
            accuracy = self.model.score(self.X_test, self.y_test)
            self.assertGreater(accuracy, 0.5)  # Assuming 50% as minimum threshold

class TestLogging(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.logger = setup_logger()
        self.log_dir = "logs"

    def test_log_directory_creation(self):
        """Test if log directory is created"""
        self.assertTrue(os.path.exists(self.log_dir))

    def test_log_file_creation(self):
        """Test if log files are created"""
        self.logger.info("Test log entry")
        log_files = os.listdir(self.log_dir)
        self.assertTrue(any(file.endswith('.log') for file in log_files))

def main():
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main()
