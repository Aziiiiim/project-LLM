FROM python:3.13.12-slim as base

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install -r ./requirements.txt

COPY ./app.py ./app.py
COPY ./code ./code

CMD ["python", "-m", "gunicorn", "-b", "0.0.0.0:8000", "app:app"]

