# -----------------------------
# Base Image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# System Settings
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# -----------------------------
# Install system dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install Python dependencies
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy project files
# -----------------------------
COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY data/ data/

# -----------------------------
# Expose Streamlit Port
# -----------------------------
EXPOSE 8501

# -----------------------------
# Run Streamlit App
# -----------------------------
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
