FROM python:3.8-slim

# Install required packages
RUN pip install flask ddtrace

# Copy application code
COPY . /app
WORKDIR /app

# Set environment variables for Datadog
ENV DD_AGENT_HOST=localhost
ENV DD_ENV=production
ENV DD_SERVICE=flask_app
ENV DD_VERSION=1.0.0

# Run the application with ddtrace
CMD ["ddtrace-run", "python", "app.py"]
