FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /app
COPY requirements.tx ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app.py ./

COPY ./models/count_vectorizer.joblib ./models/count_vectorizer.joblib
COPY ./models/naivebayes_clf.joblib ./models/naivebayes_clf.joblib

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]