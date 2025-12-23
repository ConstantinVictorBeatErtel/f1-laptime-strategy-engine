# Use a lightweight Python Linux image
FROM python:3.9-slim

WORKDIR /app

# Install system libraries needed for XGBoost/Postgres
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python tools
RUN pip install streamlit pandas psycopg2-binary mlflow==2.7.1 xgboost sqlalchemy

# Copy the app code
COPY app.py .

# Streamlit runs on port 8501
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]