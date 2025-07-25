FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app .

VOLUME "/root/.cache/huggingface"

EXPOSE 8000

HEALTHCHECK --interval=60s --start-period=60s \
    CMD curl -H "Authorization: Bearer $API_TOKEN" -f http://localhost:8000/health \
    || exit 1
    
# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]