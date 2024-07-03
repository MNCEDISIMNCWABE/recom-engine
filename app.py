from flask import Flask, jsonify, request
from recommendation import generate_recommendations, get_last_played_game
import logging
from logging.handlers import RotatingFileHandler
from ddtrace import patch_all, tracer, config
from datadog import statsd, initialize, api
from flasgger import Swagger
import json_log_formatter

# Datadog API key and application key configuration for testing
options = {
    'api_key': '99ff0fb1ea7215302a0338860fa9d373',
    'app_key': '46a2b52041682068d03a8b49b152083b209141b5'
}
initialize(**options)

# Enable Datadog tracing
patch_all()

# Set the Datadog tracer configuration to the ClusterIP service of Datadog agent
config.tracer.hostname = 'datadog-agent.default.svc.cluster.local'
config.tracer.port = 8126

# Initialize Flask app
app = Flask(__name__)
swagger = Swagger(app)  # Initialize Swagger

# Configure JSON logging
formatter = json_log_formatter.JSONFormatter()
json_handler = logging.FileHandler(filename='/app/logs/my-log.json')
json_handler.setFormatter(formatter)
logger = logging.getLogger('my_json')
logger.addHandler(json_handler)
logger.setLevel(logging.INFO)

# Set Datadog APM environment and service name
config.env = "production"
config.service = "games-recom-test"

# Initialize Datadog StatsD
statsd.constant_tags = ["env:production"]

# Define middleware for Datadog tracing
@app.before_request
def add_tracing():
    span = tracer.trace("recom_test.requests", service="flask-app", resource=request.endpoint)
    span.set_tag("http.method", request.method)
    span.set_tag("http.url", request.url)
    statsd.increment('recom_test.requests', tags=[f"endpoint:{request.endpoint}", f"method:{request.method}"])

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
            statsd.increment('recom_test.error', tags=[f"endpoint:{request.endpoint}", f"method:{request.method}"])
        span.finish()

@tracer.wrap(name='generate_recommendations', service='games-recom-test')
def wrapped_generate_recommendations():
    return generate_recommendations()

@tracer.wrap(name='get_last_played_game', service='games-recom-test')
def wrapped_get_last_played_game(user_id):
    return get_last_played_game(user_id)

try:
    recommendations_df = wrapped_generate_recommendations()
except Exception as e:
    logger.error(f"Error generating recommendations: {e}")
    recommendations_df = None

@app.route('/')
@tracer.wrap()
def index():
    """Landing page route.
    ---
    get:
        description: Welcome to the Flask API integrated with Datadog.
        responses:
            200:
                description: Returns a welcome message and a successful API response from Datadog.
    """
    logger.info('Index route accessed')
    statsd.increment('index.page_views')
    response = api.Metric.send(
        metric='recom_test_app.request_count',
        points=1,
        tags=["app:flask", "environment:production"]
    )
    logger.debug(f'Datadog API response: {response}')
    return jsonify({"message": "Welcome to the Flask API with Datadog integration!"})

@app.route('/recommend', methods=['POST'])
@tracer.wrap()
def recommend():
    """Post endpoint to generate game recommendations.
    ---
    post:
        description: Uses the user_id to fetch and generate game recommendations.
        parameters:
          - in: body
            name: body
            description: JSON object containing the user_id.
            required: true
            schema:
              type: object
              properties:
                user_id:
                  type: string
                  example: "user +27732819038"
        responses:
            200:
                description: Returns game recommendations.
            400:
                description: Error if user_id is not provided.
            404:
                description: Error if no data is found for the user_id.
            500:
                description: Internal server error.
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', '')

        if not user_id:
            statsd.increment('recom_test.error', tags=["type:missing_user_id"])
            return jsonify({"error": "User ID must be provided"}), 400

        if recommendations_df is None:
            statsd.increment('recom_test.error', tags=["type:no_recommendations"])
            return jsonify({"error": "No recommendations generated."}), 500

        last_played_game = wrapped_get_last_played_game(user_id)
        if not last_played_game:
            logger.error(f"No last played game found for user '{user_id}'")
            statsd.increment('recom_test.error', tags=["type:no_last_played_game"])
            return jsonify({"error": f"No last played game found for user '{user_id}'"}), 404

        user_recommendations = recommendations_df[recommendations_df['user_id'] == user_id]
        if user_recommendations.empty:
            logger.error(f"No recommendations found for user '{user_id}'")
            statsd.increment('recom_test.error', tags=["type:no_recommendations_for_user"])
            return jsonify({"error": f"No recommendations found for user '{user_id}'"}), 404

        recommendations = user_recommendations.to_dict(orient='records')
        return jsonify({
            "last_played_game": last_played_game,
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error in recommendation: {e}")
        statsd.increment('recom_test.error', tags=["type:internal_error"])
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
