FROM python:3.12-alpine
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./app /app
ENV PORT=80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
