import os
import re
import string
import unicodedata
import logging
import time

import contractions
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from google.cloud import bigquery
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pandas_gbq import to_gbq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from textblob import TextBlob
from unidecode import unidecode
from google.cloud import monitoring_v3

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'bright-arc-328707-b5e2d782b48b.json'

# Initialize the BigQuery client
client = bigquery.Client()

# Initialize Google Cloud Monitoring client
monitoring_client = monitoring_v3.MetricServiceClient()
project_name = f"projects/{client.project}"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_data(path_to_csv_file):
    try:
        df = pd.read_csv(path_to_csv_file)
        return df
    except Exception as e:
        logger.error(f"Error reading data from {path_to_csv_file}: {e}")
        raise

# Load data
try:
    df_user_last_game_played = read_data('last_played_game.csv')
    df_all_available_games = read_data('all_games.csv')
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise

class NltkPreprocessingSteps:
    def __init__(self, X):
        self.X = X
        self.sw_nltk = stopwords.words('english')
        new_stopwords = ['<*>','Ayoba','ayoba']
        self.sw_nltk.extend(new_stopwords)
        self.remove_punctuations = string.punctuation.replace('.','')

    def remove_html_tags(self):
        self.X = self.X.apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
        return self

    def remove_accented_chars(self):
        self.X = self.X.apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore'))
        return self

    def replace_diacritics(self):
        self.X = self.X.apply(lambda x: unidecode(x, errors="preserve"))
        return self

    def to_lower(self):
        self.X = self.X.apply(lambda x: " ".join([word.lower() for word in x.split() if word and word not in self.sw_nltk]) if x else '')
        return self

    def expand_contractions(self):
        self.X = self.X.apply(lambda x: " ".join([contractions.fix(expanded_word) for expanded_word in x.split()]))
        return self

    def remove_numbers(self):
        self.X = self.X.apply(lambda x: re.sub(r'\d+', '', x))
        return self

    def remove_http(self):
        self.X = self.X.apply(lambda x: re.sub(r'http\S+', '', x))
        return self
    
    def remove_words_with_numbers(self):
        self.X = self.X.apply(lambda x: re.sub(r'\w*\d\w*', '', x))
        return self
    
    def remove_digits(self):
        self.X = self.X.apply(lambda x: re.sub(r'[0-9]+', '', x))
        return self
    
    def remove_special_character(self):
        self.X = self.X.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]+', ' ', x))
        return self
    
    def remove_white_spaces(self):
        self.X = self.X.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        return self
    
    def remove_extra_newlines(self):
        self.X == self.X.apply(lambda x: re.sub(r'[\r|\n|\r\n]+', ' ', x))
        return self

    def replace_dots_with_spaces(self):
        self.X = self.X.apply(lambda x: re.sub("[.]", " ", x))
        return self

    def remove_punctuations_except_periods(self):
        self.X = self.X.apply(lambda x: re.sub('[%s]' % re.escape(self.remove_punctuations), '' , x))
        return self

    def remove_all_punctuations(self):
        self.X = self.X.apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))
        return self

    def remove_double_spaces(self):
        self.X = self.X.apply(lambda x: re.sub(' +', '  ', x))
        return self

    def fix_typos(self):
        self.X = self.X.apply(lambda x: str(TextBlob(x).correct()))
        return self

    def remove_stopwords(self):
        self.X = self.X.apply(lambda x: " ".join([ word for word in x.split() if word not in self.sw_nltk]))
        return self
    
    def remove_singleChar(self):
        self.X = self.X.apply(lambda x: " ".join([ word for word in x.split() if len(word)>2]))
        return self

    def lemmatize(self):
        lemmatizer = WordNetLemmatizer()
        self.X = self.X.apply(lambda x: " ".join([ lemmatizer.lemmatize(word) for word in x.split()]))
        return self

    def get_processed_text(self):
        return self.X

# Preprocess data
try:
    txt_preproc_all_games = NltkPreprocessingSteps(df_all_available_games['game_title'])
    txt_preproc_user_games = NltkPreprocessingSteps(df_user_last_game_played['game'])

    processed_text_all_games = txt_preproc_all_games.to_lower().remove_html_tags().remove_accented_chars().replace_diacritics().expand_contractions().remove_numbers().remove_digits().remove_special_character().remove_white_spaces().remove_extra_newlines().replace_dots_with_spaces().remove_punctuations_except_periods().remove_words_with_numbers().remove_singleChar().remove_double_spaces().lemmatize().remove_stopwords().get_processed_text()
    processed_text_all_users = txt_preproc_user_games.to_lower().remove_html_tags().remove_accented_chars().replace_diacritics().expand_contractions().remove_numbers().remove_digits().remove_special_character().remove_white_spaces().remove_extra_newlines().replace_dots_with_spaces().remove_punctuations_except_periods().remove_words_with_numbers().remove_singleChar().remove_double_spaces().lemmatize().remove_stopwords().get_processed_text()

    df_all_available_games['game_title_processed'] = processed_text_all_games
    df_user_last_game_played['game_processed'] = processed_text_all_users
except Exception as e:
    logger.error(f"Error in data preprocessing: {e}")
    raise

# TF-IDF vectorization
try:
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_all_available_games['game_title_processed'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
except Exception as e:
    logger.error(f"Error in TF-IDF vectorization: {e}")
    raise

def get_content_based_recommendations(game_name, cosine_sim=cosine_sim, df_all_available_games=df_all_available_games):
    try:
        idx = df_all_available_games[df_all_available_games['game_title_processed'] == game_name].index
        if len(idx) == 0:
            return []
        else:
            idx = idx[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]  # Top 10 similar games
            similar_games = [df_all_available_games['game_title_processed'][i[0]] for i in sim_scores]
            return similar_games
    except Exception as e:
        logger.error(f"Error in getting content-based recommendations: {e}")
        raise

def record_metric(metric_name, value):
    series = monitoring_v3.TimeSeries()
    series.metric.type = f"custom.googleapis.com/{metric_name}"
    series.resource.type = "global"

    # Create a new point
    point = monitoring_v3.Point()
    point.value.double_value = value

    now = time.time()
    seconds = int(now)
    nanos = int((now - seconds) * 10**9)
    interval = monitoring_v3.TimeInterval({
        "end_time": {"seconds": seconds, "nanos": nanos}
    })
    point.interval = interval

    # Add the point to the time series
    series.points = [point]

    # Create the time series in Cloud Monitoring
    monitoring_client.create_time_series(name=project_name, time_series=[series])

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        user_id = data.get('user_id', '')

        if not user_id:
            logger.error('recommendation_errors: User ID must be provided')
            record_metric('recommendation_errors', 1)
            return jsonify({"error": "User ID must be provided"}), 400

        # Retrieve last played game for the user
        user_game = df_user_last_game_played[df_user_last_game_played['user_id'] == user_id]['game_processed'].values
        if len(user_game) == 0:
            logger.error(f'recommendation_errors: No last played game found for user {user_id}')
            record_metric('recommendation_errors', 1)
            return jsonify({"error": f"No last played game found for user '{user_id}'"}), 404

        game_name = user_game[0]

        # Get recommendations
        idx = df_all_available_games[df_all_available_games['game_title_processed'] == game_name].index
        if len(idx) == 0:
            logger.error(f'recommendation_errors: No similar games found for the last played game {game_name}')
            record_metric('recommendation_errors', 1)
            return jsonify({"error": f"No similar games found for the last played game '{game_name}'"}), 404

        idx = idx[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]  # Top 10

        recommendations = []
        for rank, i in enumerate(sim_scores, start=1):
            game_info = df_all_available_games.iloc[i[0]]
            recommendations.append({
                "user_id": int(user_id),
                "game_id": int(game_info['game_id']),
                "title": game_info['game_title'],
                "country": game_info.get('country', ''),
                "city": game_info.get('city', ''),
                "City_Ranking": rank,  # Rank from 1 to 10
                "recommendation_type": "games",
                "recommendation_activity": "user_activity"
            })

        logger.info('recommendation_success: Successfully generated recommendations')
        record_metric('recommendation_successes', 1)
        return jsonify({
            "last_played_game": game_name,
            "recommendations": recommendations
        })

    except Exception as e:
        logger.error(f'recommendation_errors: Error in recommendation: {e}')
        record_metric('recommendation_errors', 1)
        return jsonify({"error": "Internal server error"}), 500

def create_and_load_recommendations(df):
    schema = [
        bigquery.SchemaField("user_id", "STRING"),
        bigquery.SchemaField("country", "STRING"),
        bigquery.SchemaField("city", "STRING"),
        bigquery.SchemaField("GameID", "INTEGER"),
        bigquery.SchemaField("City_Ranking", "INTEGER"),
        bigquery.SchemaField("recommendation_type", "STRING"),
        bigquery.SchemaField("recommendation_activity", "STRING"),
    ]

    dataset_id = 'bright-arc-328707.ayoba' 
    table_id = f'{dataset_id}.rec_games_recommendations_activity_staging'
    table = bigquery.Table(table_id, schema=schema)

    try:
        client.delete_table(table)
        print(f"Dropped table {table_id}")
    except Exception as e:
        logger.warning(f"Table {table_id} does not exist: {e}")

    table.clustering_fields = ["user_id"]
    table = client.create_table(table, exists_ok=True)
    print(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")

    df['GameID'] = df['GameID'].fillna(-1).astype(np.int64)
    df['City_Ranking'] = df['City_Ranking'].fillna(-1).astype(np.int64)
    df['user_id'] = df['user_id'].astype(str)
    df['country'] = df['country'].astype(str)
    df['city'] = df['city'].astype(str)
    df['recommendation_type'] = df['recommendation_type'].astype(str)
    df['recommendation_activity'] = df['recommendation_activity'].astype(str)

    to_gbq(df, table_id, project_id=table.project, if_exists='append')
    print('Data loading done')

def manage_down_stream_update():
    merge_sql = """
    MERGE `bright-arc-328707.ayoba.rec_card_recommendations` AS A
    USING (SELECT * FROM `bright-arc-328707.ayoba.rec_games_recommendations_activity_staging` 
            where GameID is not null or GameID !=-1)  AS B
    ON (A.user_id = B.user_id 
        AND A.GameID = B.GameID
        AND B.recommendation_type = B.recommendation_type
        AND B.recommendation_activity)
    WHEN MATCHED AND A.City_Ranking != B.City_Ranking THEN
      UPDATE SET 
        A.City_Ranking = B.City_Ranking
    WHEN NOT MATCHED BY TARGET THEN
      INSERT (user_id, country, city, GameID, City_Ranking, recommendation_type, recommendation_activity)
      VALUES (B.user_id, B.country, B.city, B.GameID, B.City_Ranking, B.recommendation_type, B.recommendation_activity)
    WHEN NOT MATCHED BY SOURCE 
        AND recommendation_type = 'user_activity' 
        AND recommendation_type = 'games' THEN
      DELETE
    """
    query_job = client.query(merge_sql)
    query_job.result()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
