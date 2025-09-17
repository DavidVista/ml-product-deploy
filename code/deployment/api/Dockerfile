# FastAPI Dockerfile
FROM python:3.13-alpine

# Set the working directory
WORKDIR /api

# Copy and install requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy api.py and model files
COPY api.py .
RUN mkdir models
COPY ../../../models /models

# Expose port 80 to the outside world
EXPOSE 80


# Run the FastAPI server
CMD ["uvicorn", "api:api", "--host", "127.0.0.1", "--port", "80"]
