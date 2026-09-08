import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    words = word_tokenize(text)
    filtered = [w for w in words if w not in stop_words]
    return " ".join(lemmatizer.lemmatize(w) for w in filtered)


def load_dataset():
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")
    df_fake["label"] = 0
    df_true["label"] = 1

    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = shuffle(df, random_state=42).reset_index(drop=True)

    df["combined"] = (df["title"] + " " + df["text"]).str.lower()
    df = df[df["combined"].str.strip() != ""]
    df["combined"] = df["combined"].apply(clean_text)
    return df


def main():
    df = load_dataset()
    df_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    # Text-only TF-IDF (unlike prep.py, this doesn't include the "date" field —
    # see README note on the known date-format leakage in this dataset).
    vectorizer = TfidfVectorizer(max_features=5000)
    x_train = vectorizer.fit_transform(df_train["combined"])
    x_test = vectorizer.transform(df_test["combined"])
    y_train = df_train["label"].values
    y_test = df_test["label"].values

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "SVM (Linear)": LinearSVC(random_state=42),
    }

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))


if __name__ == "__main__":
    main()
