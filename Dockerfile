# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements 
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose port
EXPOSE 8080

# Set environment variables for Datadog
ENV DD_AGENT_HOST=localhost
ENV DD_ENV=production
ENV DD_SERVICE=flask_app
ENV DD_VERSION=1.0.0

# Run the application with ddtrace
CMD ["ddtrace-run", "gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
