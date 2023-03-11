from fastapi import FastAPI, HTTPException
from joblib import load
from preprocess import clean_text

vectorizer = load(open("./models/count_vectorizer.joblib", "rb"))
clf = load(open("./models/naivebayes_clf.joblib", "rb"))

app = FastAPI()

@app.get("/")
def root():
    return {
        "message": "Welcome to Spam Detection API"
    }

@app.post("/predict")
def predict(text):
    prediction = ""
    if not(text):
        raise HTTPException(
            status_code=400,
            detail="Please Provide a Valid Text Message"
        )
    text = clean_text(text)
    pred = clf.predict(vectorizer.transform([text]))

    if pred[0] == 0:
        prediction = "Ham"
    elif pred[0] == 1:
        prediction = "Spam"

    return {
        "text": text,
        "prediction": prediction
    }