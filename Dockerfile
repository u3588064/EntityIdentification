FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY main.py .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "entity_comparison_server.py"]
