FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
