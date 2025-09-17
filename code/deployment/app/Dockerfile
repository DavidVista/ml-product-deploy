# Streamlit Dockerfile
FROM python:3-slim

# Set the working directory
WORKDIR /app

# Copy and install requirements
COPY code/deployment/app/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy app.py
COPY code/deployment/app/app.py .

# Expose the default Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
