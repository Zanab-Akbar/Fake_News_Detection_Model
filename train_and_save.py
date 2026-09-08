import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from svm_comparison import load_dataset


def main():
    df = load_dataset()
    df_train, _ = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    x_train = vectorizer.fit_transform(df_train["combined"])
    y_train = df_train["label"].values

    model = LogisticRegression(max_iter=5000, random_state=42)
    model.fit(x_train, y_train)

    joblib.dump(model, "model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")
    print("Saved model.joblib and vectorizer.joblib")


if __name__ == "__main__":
    main()
