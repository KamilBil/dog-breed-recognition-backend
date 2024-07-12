FROM python:3.11-slim
WORKDIR /app

# download the dataset first so as not to overload the server
COPY downloader.py .
RUN pip install requests
RUN python downloader.py
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=80
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
