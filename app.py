from flask import Flask, jsonify, request
from recommendation import generate_recommendations, get_last_played_game
import logging
from json_log_formatter import JSONFormatter
from ddtrace import tracer, patch_all
import threading

# Enable tracing
patch_all()

app = Flask(__name__)

# Configure logging
formatter = JSONFormatter()
json_handler = logging.FileHandler(filename='/var/log/my-log.json')
json_handler.setFormatter(formatter)
logger = logging.getLogger('my_json')
logger.addHandler(json_handler)
logger.setLevel(logging.INFO)

# Initialize recommendations dataframe
try:
    recommendations_df = generate_recommendations()
except Exception as e:
    app.logger.error(f"Error generating recommendations: {e}")
    recommendations_df = None

@tracer.wrap(name='flask.request')
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        user_id = str(data.get('user_id', '')).strip().lstrip('+')

        if not user_id:
            logger.error('User ID must be provided')
            return jsonify({"error": "User ID must be provided"}), 400

        # Convert user_id to int for comparison
        try:
            user_id = int(user_id)
        except ValueError:
            logger.error('Invalid User ID format')
            return jsonify({"error": "Invalid User ID format"}), 400

        last_played_game = get_last_played_game(user_id)
        if not last_played_game:
            logger.error(f"No last played game found for user '{user_id}'")
            return jsonify({"error": f"No last played game found for user '{user_id}'"}), 404

        user_recommendations = recommendations_df[recommendations_df['user_id'] == user_id]
        if user_recommendations.empty:
            logger.error(f"No recommendations found for user '{user_id}'")
            return jsonify({"error": f"No recommendations found for user '{user_id}'"}), 404

        recommendations = user_recommendations.to_dict(orient='records')
        return jsonify({
            "last_played_game": last_played_game,
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error in recommendation: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
