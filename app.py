from flask import Flask, Blueprint, jsonify, request
from recommendation import generate_recommendations, get_last_played_game
import logging
from logging.handlers import RotatingFileHandler
from ddtrace import tracer, patch_all, config
import statsd

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

# Set Datadog APM environment and service name
config.env = "production"
config.service = "games-reco-test"

# Initialize StatsD client
statsd_client = statsd.StatsClient('localhost', 8125)

# Initialize the Blueprint
bp = Blueprint('main', __name__)

@tracer.wrap(name='generate_recommendations', service='games-reco-test')
def wrapped_generate_recommendations():
    return generate_recommendations()

@tracer.wrap(name='get_last_played_game', service='games-reco-test')
def wrapped_get_last_played_game(user_id):
    return get_last_played_game(user_id)

try:
    recommendations_df = wrapped_generate_recommendations()
except Exception as e:
    app.logger.error(f"Error generating recommendations: {e}")
    recommendations_df = None

@bp.route('/', methods=['GET'])
def index():
    app.logger.info("Index route accessed")
    statsd_client.increment('index.page_views')
    return jsonify({"message": "Welcome to the Flask API with Datadog integration!"})

@bp.route('/health', methods=['GET'])
def health():
    app.logger.info("Health route accessed")
    statsd_client.increment('health.checks')
    return jsonify({"status": "ok"})

@bp.route('/recommend', methods=['POST'])
def recommend():
    try:
        app.logger.info("Recommend route accessed")
        statsd_client.increment('recommend.requests')

        if recommendations_df is None:
            return jsonify({"error": "No recommendations generated."}), 500

        data = request.get_json()
        user_id = str(data.get('user_id', '')).strip().lstrip('+')

        if not user_id:
            return jsonify({"error": "User ID must be provided"}), 400

        try:
            user_id = int(user_id)
        except ValueError:
            return jsonify({"error": "Invalid User ID format"}), 400

        last_played_game = wrapped_get_last_played_game(user_id)
        if not last_played_game:
            app.logger.error(f"No last played game found for user '{user_id}'")
            return jsonify({"error": f"No last played game found for user '{user_id}'"}), 404

        user_recommendations = recommendations_df[recommendations_df['user_id'] == str(user_id)]
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
        statsd_client.increment('recommend.errors')
        return jsonify({"error": "Internal server error"}), 500

@app.before_request
def start_trace():
    span = tracer.trace("flask.request", service="flask-app", resource=request.endpoint)
    span.set_tag("http.method", request.method)
    span.set_tag("http.url", request.url)

@app.after_request
def stop_trace(response):
    span = tracer.current_span()
    if span:
        span.set_tag("http.status_code", response.status_code)
        span.finish()
    return response

@app.teardown_request
def teardown_trace(exception):
    span = tracer.current_span()
    if span:
        if exception:
            span.set_tag("error", str(exception))
            span.set_tag("http.status_code", 500)
        span.finish()

# Register the blueprint
app.register_blueprint(bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
