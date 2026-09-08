import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = FastAPI(title="Fake News Detector")

model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


class Article(BaseModel):
    title: str
    text: str


def clean_text(text: str) -> str:
    words = word_tokenize(text.lower())
    filtered = [w for w in words if w not in stop_words]
    return " ".join(lemmatizer.lemmatize(w) for w in filtered)


@app.get("/")
def root():
    return {"message": "POST /predict with {title, text} to classify an article."}


@app.post("/predict")
def predict(article: Article):
    combined = clean_text(f"{article.title} {article.text}")
    features = vectorizer.transform([combined])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][int(prediction)]
    return {
        "label": "Real" if prediction == 1 else "Fake",
        "confidence": round(float(probability), 3),
    }
