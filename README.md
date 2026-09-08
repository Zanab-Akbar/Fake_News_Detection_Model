# Fake News Detection Model 📰

An NLP project that classifies news articles as real or fake, built on the classic Kaggle "Fake and Real News" dataset.

## Approach

- **Preprocessing** (`prep.py`, `fake.ipynb`): lowercase, drop blank rows, tokenize, remove stopwords, and lemmatize article title+text using NLTK
- **Feature extraction**: TF-IDF vectorization (scikit-learn `TfidfVectorizer`) over the cleaned text and the article date
- **Model**: Logistic Regression (scikit-learn), trained on an 80/20 stratified train/test split
- **Evaluation**: accuracy, precision/recall/F1 (classification report), confusion matrix, and ROC curve/AUC

## Results

- **87.9% accuracy** on the held-out test set (~9,000 articles)
- Balanced precision/recall (~0.87–0.89) across both "Fake" and "Real" classes

## Tech Stack

Python, pandas, NLTK, scikit-learn, matplotlib, Jupyter

## Project Structure

- `fake.ipynb` — main notebook: full pipeline from raw CSVs to trained model and evaluation plots
- `prep.py` — standalone script version of the preprocessing + training pipeline
- `p.py` — a separate exploratory script tallying rows across `train.tsv`/`test.tsv`/`valid.tsv` (a differently-formatted dataset variant, not part of the main pipeline above)

## How to Run

1. Download the [Fake and Real News dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle and place `Fake.csv` and `True.csv` in this folder
2. Install dependencies: `pip install pandas nltk scikit-learn matplotlib scipy`
3. Run `python prep.py` or open `fake.ipynb` in Jupyter to see the full pipeline and evaluation plots

## Planned Next Steps

- Add a FastAPI inference endpoint for serving predictions
- Compare Logistic Regression against an SVM classifier
