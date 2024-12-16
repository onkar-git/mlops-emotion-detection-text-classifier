FROM python:3.9

WORKDIR /app

COPY flask_app/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

RUN pip install -r requirement.txt

EXPOSE 5000

CMD  ["python","app.py"]
