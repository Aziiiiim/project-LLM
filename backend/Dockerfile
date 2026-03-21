FROM python:3.13-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . .
RUN python -m pip install -r ./requirements.txt

CMD ["python", "-m", "gunicorn", "-b", "0.0.0.0:8000", "--timeout", "3600", "app:app"]

