import os
import re
import string
import unicodedata
import contractions
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from google.cloud import bigquery
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
from textblob import TextBlob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Set up Google Cloud credentials
try:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'ornate-genre-425416-q8-39e4e509df0e.json'
except Exception as e:
    logger.error("Error setting GOOGLE_APPLICATION_CREDENTIALS: %s", e)

# Initialize the BigQuery client
try:
    client = bigquery.Client()
except Exception as e:
    logger.error("Error initializing BigQuery client: %s", e)

def read_data(path_to_csv_file):
    try:
        df = pd.read_csv(path_to_csv_file)
        return df
    except Exception as e:
        logger.error(f"Error reading data from {path_to_csv_file}: {e}")

# Load data
try:
    df_user_last_game_played = read_data('last_played_game.csv')
    df_all_available_games = read_data('all_games.csv')
except Exception as e:
    logger.error(f"Error loading data: {e}")

class NltkPreprocessingSteps:
    def __init__(self, X):
        self.X = X
        self.sw_nltk = stopwords.words('english')
        new_stopwords = ['<*>','Ayoba','ayoba']
        self.sw_nltk.extend(new_stopwords)
        self.remove_punctuations = string.punctuation.replace('.','')

    def remove_html_tags(self):
        try:
            self.X = self.X.apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
            return self
        except Exception as e:
            logger.error(f"Error removing HTML tags: {e}")

    def remove_accented_chars(self):
        try:
            self.X = self.X.apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore'))
            return self
        except Exception as e:
            logger.error(f"Error removing accented characters: {e}")

    def replace_diacritics(self):
        try:
            self.X = self.X.apply(lambda x: unidecode(x, errors="preserve"))
            return self
        except Exception as e:
            logger.error(f"Error replacing diacritics: {e}")

    def to_lower(self):
        try:
            self.X = self.X.apply(lambda x: " ".join([word.lower() for word in x.split() if word and word not in self.sw_nltk]) if x else '')
            return self
        except Exception as e:
            logger.error(f"Error converting to lower case: {e}")

    def expand_contractions(self):
        try:
            self.X = self.X.apply(lambda x: " ".join([contractions.fix(expanded_word) for expanded_word in x.split()]))
            return self
        except Exception as e:
            logger.error(f"Error expanding contractions: {e}")

    def remove_numbers(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'\d+', '', x))
            return self
        except Exception as e:
            logger.error(f"Error removing numbers: {e}")

    def remove_http(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'http\S+', '', x))
            return self
        except Exception as e:
            logger.error(f"Error removing http links: {e}")
    
    def remove_words_with_numbers(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'\w*\d\w*', '', x))
            return self
        except Exception as e:
            logger.error(f"Error removing words with numbers: {e}")
    
    def remove_digits(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'[0-9]+', '', x))
            return self
        except Exception as e:
            logger.error(f"Error removing digits: {e}")
    
    def remove_special_character(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]+', ' ', x))
            return self
        except Exception as e:
            logger.error(f"Error removing special characters: {e}")
    
    def remove_white_spaces(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
            return self
        except Exception as e:
            logger.error(f"Error removing white spaces: {e}")
    
    def remove_extra_newlines(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'[\r|\n|\r\n]+', ' ', x))
            return self
        except Exception as e:
            logger.error(f"Error removing extra newlines: {e}")

    def replace_dots_with_spaces(self):
        try:
            self.X = self.X.apply(lambda x: re.sub("[.]", " ", x))
            return self
        except Exception as e:
            logger.error(f"Error replacing dots with spaces: {e}")

    def remove_punctuations_except_periods(self):
        try:
            self.X = self.X.apply(lambda x: re.sub('[%s]' % re.escape(self.remove_punctuations), '' , x))
            return self
        except Exception as e:
            logger.error(f"Error removing punctuations except periods: {e}")

    def remove_all_punctuations(self):
        try:
            self.X = self.X.apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))
            return self
        except Exception as e:
            logger.error(f"Error removing all punctuations: {e}")

    def remove_double_spaces(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(' +', '  ', x))
            return self
        except Exception as e:
            logger.error(f"Error removing double spaces: {e}")

    def fix_typos(self):
        try:
            self.X = self.X.apply(lambda x: str(TextBlob(x).correct()))
            return self
        except Exception as e:
            logger.error(f"Error fixing typos: {e}")

    def remove_stopwords(self):
        try:
            self.X = self.X.apply(lambda x: " ".join([ word for word in x.split() if word not in self.sw_nltk]))
            return self
        except Exception as e:
            logger.error(f"Error removing stopwords: {e}")
    
    def remove_singleChar(self):
        try:
            self.X = self.X.apply(lambda x: " ".join([ word for word in x.split() if len(word)>2]))
            return self
        except Exception as e:
            logger.error(f"Error removing single characters: {e}")

    def lemmatize(self):
        try:
            lemmatizer = WordNetLemmatizer()
            self.X = self.X.apply(lambda x: " ".join([ lemmatizer.lemmatize(word) for word in x.split()]))
            return self
        except Exception as e:
            logger.error(f"Error lemmatizing: {e}")

    def get_processed_text(self):
        return self.X

try:
    txt_preproc_all_games = NltkPreprocessingSteps(df_all_available_games['game_title'])
    txt_preproc_user_games = NltkPreprocessingSteps(df_user_last_game_played['game'])

    processed_text_all_games = txt_preproc_all_games.to_lower().remove_html_tags().remove_accented_chars().replace_diacritics().expand_contractions().remove_numbers().remove_digits().remove_special_character().remove_white_spaces().remove_extra_newlines().replace_dots_with_spaces().remove_punctuations_except_periods().remove_words_with_numbers().remove_singleChar().remove_double_spaces().lemmatize().remove_stopwords().get_processed_text()
    processed_text_all_users = txt_preproc_user_games.to_lower().remove_html_tags().remove_accented_chars().replace_diacritics().expand_contractions().remove_numbers().remove_digits().remove_special_character().remove_white_spaces().remove_extra_newlines().replace_dots_with_spaces().remove_punctuations_except_periods().remove_words_with_numbers().remove_singleChar().remove_double_spaces().lemmatize().remove_stopwords().get_processed_text()

    df_all_available_games['game_title_processed'] = processed_text_all_games
    df_user_last_game_played['game_processed'] = processed_text_all_users
except Exception as e:
    logger.error(f"Error in preprocessing: {e}")
