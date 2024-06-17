import os
import re
import string
import unicodedata

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

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/Users/mncedisimncwabe/Documents/Personal_Google_Service_Account/bright-arc-328707-b5e2d782b48b.json'

# Initialize the BigQuery client
client = bigquery.Client()

def read_data(path_to_csv_file):
    df = pd.read_csv(path_to_csv_file)
    return df

# Load data
df_user_last_game_played = read_data('last_played_game.csv')
df_all_available_games = read_data('all_games.csv')

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
txt_preproc_all_games = NltkPreprocessingSteps(df_all_available_games['game_title'])
txt_preproc_user_games = NltkPreprocessingSteps(df_user_last_game_played['game'])

processed_text_all_games = txt_preproc_all_games.to_lower().remove_html_tags().remove_accented_chars().replace_diacritics().expand_contractions().remove_numbers().remove_digits().remove_special_character().remove_white_spaces().remove_extra_newlines().replace_dots_with_spaces().remove_punctuations_except_periods().remove_words_with_numbers().remove_singleChar().remove_double_spaces().lemmatize().remove_stopwords().get_processed_text()
processed_text_all_users = txt_preproc_user_games.to_lower().remove_html_tags().remove_accented_chars().replace_diacritics().expand_contractions().remove_numbers().remove_digits().remove_special_character().remove_white_spaces().remove_extra_newlines().replace_dots_with_spaces().remove_punctuations_except_periods().remove_words_with_numbers().remove_singleChar().remove_double_spaces().lemmatize().remove_stopwords().get_processed_text()

df_all_available_games['game_title_processed'] = processed_text_all_games
df_user_last_game_played['game_processed'] = processed_text_all_users

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_all_available_games['game_title_processed'])

# Compute similarity scores
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_content_based_recommendations(game_name, cosine_sim=cosine_sim, df_all_available_games=df_all_available_games):
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

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    game_name = data.get('game_name', '')
    user_id = data.get('user_id', '')

    if not game_name or not user_id:
        return jsonify({"error": "Both game name and user ID must be provided"}), 400

    # Preprocess the game name
    processed_game_name = NltkPreprocessingSteps(pd.Series([game_name])).to_lower().remove_html_tags().remove_accented_chars().replace_diacritics().expand_contractions().remove_numbers().remove_digits().remove_special_character().remove_white_spaces().remove_extra_newlines().replace_dots_with_spaces().remove_punctuations_except_periods().remove_words_with_numbers().remove_singleChar().remove_double_spaces().lemmatize().remove_stopwords().get_processed_text().iloc[0]

    idx = df_all_available_games[df_all_available_games['game_title_processed'] == processed_game_name].index
    if len(idx) == 0:
        return jsonify({"error": f"No similar games found for '{game_name}'"}), 404

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

    return jsonify({
        "game_name": game_name,
        "recommendations": recommendations
    })

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
    except:
        pass

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