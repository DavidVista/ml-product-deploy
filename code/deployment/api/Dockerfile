# FastAPI Dockerfile
FROM python:3-slim

# Set the working directory
WORKDIR /api

# Copy and install requirements
COPY code/deployment/api/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy api.py and model files
COPY code/deployment/api/api.py .
COPY models/ ./models

# Expose port 80 to the outside world
EXPOSE 80


# Run the FastAPI server
CMD ["uvicorn", "api:api", "--host", "0.0.0.0", "--port", "80"]
