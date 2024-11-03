FROM python:3.11-slim
WORKDIR /app

# download the dataset first so as not to overload the server
COPY downloader.py .
RUN pip install requests
RUN python downloader.py

COPY . .
RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi
ENV PORT=80
CMD ["poetry", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
