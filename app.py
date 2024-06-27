from ddtrace import tracer, patch_all
from flask import Flask, jsonify, request
import logging

patch_all()
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

@app.route('/recommend', methods=['POST'])
def recommend():
    with tracer.trace("flask.request", service="flask-app"):
        try:
            if recommendations_df is None:
                return jsonify({"error": "Recommendations are not available at the moment"}), 500

            data = request.get_json()
            user_id = str(data.get('user_id', '')).strip().lstrip('+')

            if not user_id:
                return jsonify({"error": "User ID must be provided"}), 400

            try:
                user_id = int(user_id)
            except ValueError:
                return jsonify({"error": "Invalid User ID format"}), 400

            last_played_game = get_last_played_game(user_id)
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
