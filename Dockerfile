FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default command (you can override it in docker-compose)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
