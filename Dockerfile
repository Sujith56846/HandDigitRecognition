# Dockerfile for Flask app on Hugging Face Spaces
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_flask_hf.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app_flask.py .
COPY digit_recognition_cnn_model.pth .
COPY digit_recognition_ann_model.pth .
COPY templates/ templates/
COPY static/ static/

# Expose port 7860 (required by Hugging Face)
EXPOSE 7860

# Set environment variable for Flask
ENV FLASK_APP=app_flask.py

# Run the application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=7860"]
