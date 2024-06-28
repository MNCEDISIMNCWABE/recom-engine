from flask import Flask, jsonify, request
from recommendation import generate_recommendations, get_last_played_game
import logging
from logging.handlers import RotatingFileHandler
from ddtrace import tracer, patch_all

# Enable Datadog tracing
patch_all()

app = Flask(__name__)

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file = '/app/logs/flask.log'
file_handler = RotatingFileHandler(log_file, maxBytes=10000, backupCount=1)
file_handler.setFormatter(log_formatter)

app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

try:
    recommendations_df = generate_recommendations()
except Exception as e:
    app.logger.error(f"Error generating recommendations: {e}")
    recommendations_df = None

@tracer.wrap(name='generate_recommendations')
def wrapped_generate_recommendations():
    return generate_recommendations()

@tracer.wrap(name='get_last_played_game')
def wrapped_get_last_played_game(user_id):
    return get_last_played_game(user_id)

@app.route('/recommend', methods=['POST'])
@tracer.wrap(name='recommend')
def recommend():
    try:
        if recommendations_df is None:
            return jsonify({"error": "Recommendations are not available at the moment"}), 500

        data = request.get_json()
        user_id = str(data.get('user_id', '')).strip().lstrip('+')

        if not user_id:
            return jsonify({"error": "User ID must be provided"}), 400

        # Convert user_id to int for comparison
        try:
            user_id = int(user_id)
        except ValueError:
            return jsonify({"error": "Invalid User ID format"}), 400

        last_played_game = wrapped_get_last_played_game(user_id)
        if not last_played_game:
            app.logger.error(f"No last played game found for user '{user_id}'")
            return jsonify({"error": f"No last played game found for user '{user_id}'"}), 404

        user_recommendations = recommendations_df[recommendations_df['user_id'] == user_id]
        if user_recommendations.empty:
            app.logger.error(f"No recommendations found for user '{user_id}'")
            return jsonify({"error": f"No recommendations found for user '{user_id}'"}), 404

        recommendations = user_recommendations.to_dict(orient='records')
        return jsonify({
            "last_played_game": last_played_game,
            "recommendations": recommendations
        })
    except Exception as e:
        app.logger.error(f"Error in recommendation: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
