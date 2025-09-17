# Streamlit Dockerfile
FROM python:3.13-alpine

# Set the working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy app.py
COPY app.py .

# Expose the default Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=127.0.0.1"]
