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

# Copy Datadog configuration
COPY datadog-agent/conf.d/python.d /etc/datadog-agent/conf.d/python.d

# Create the logs directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 8080

# Run app.py with ddtrace
CMD ["ddtrace-run", "python", "app.py"]
