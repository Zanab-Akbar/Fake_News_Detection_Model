# Fake News Detection Model 📰

An NLP project that classifies news articles as real or fake, built on the classic Kaggle "Fake and Real News" dataset.

## Approach

- **Preprocessing** (`prep.py`, `fake.ipynb`, `svm_comparison.py`): lowercase, drop blank rows, tokenize, remove stopwords, and lemmatize article title+text using NLTK
- **Feature extraction**: TF-IDF vectorization (scikit-learn `TfidfVectorizer`)
- **Models**: Logistic Regression and Linear SVM (scikit-learn), trained on an 80/20 stratified train/test split
- **Evaluation**: accuracy, precision/recall/F1 (classification report), confusion matrix, and ROC curve/AUC
- **Serving**: a FastAPI endpoint (`api.py`) that loads the trained model and classifies new article text

## Results

- **87.9% accuracy** on the held-out test set (~9,000 articles) with Logistic Regression + TF-IDF
- Balanced precision/recall (~0.87–0.89) across both "Fake" and "Real" classes
- See `svm_comparison.py` for a head-to-head Logistic Regression vs. SVM comparison

**Note:** this dataset has a well-documented quirk where "Fake" and "Real" articles come from different sources with distinguishable date formatting, which can let a model perform well by picking up on that formatting rather than genuine content signal. `svm_comparison.py` trains on article text only (no date feature) to avoid leaning on this; `prep.py`/`fake.ipynb` include a date-based TF-IDF feature alongside text, which is worth keeping in mind when interpreting that pipeline's results.

## Tech Stack

Python, pandas, NLTK, scikit-learn, FastAPI, joblib, matplotlib, Jupyter

## Project Structure

- `fake.ipynb` — main notebook: full pipeline from raw CSVs to trained model and evaluation plots
- `prep.py` — standalone script version of the preprocessing + training pipeline
- `svm_comparison.py` — trains and compares Logistic Regression vs. SVM on text-only TF-IDF features
- `train_and_save.py` — trains a Logistic Regression model and saves it (`model.joblib`, `vectorizer.joblib`) for the API to serve
- `api.py` — FastAPI app exposing a `/predict` endpoint for classifying new article text
- `p.py` — a separate exploratory script tallying rows across `train.tsv`/`test.tsv`/`valid.tsv` (a differently-formatted dataset variant, not part of the main pipeline above)

## How to Run

1. Download the [Fake and Real News dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle and place `Fake.csv` and `True.csv` in this folder
2. Install dependencies: `pip install -r requirements.txt`
3. Explore the pipeline: run `python prep.py` or open `fake.ipynb` in Jupyter
4. Compare models: `python svm_comparison.py`
5. Serve predictions:
   ```bash
   python train_and_save.py   # trains and saves model.joblib + vectorizer.joblib
   uvicorn api:app --reload   # starts the API on http://127.0.0.1:8000
   ```
   Then POST to `/predict`:
   ```bash
   curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"title": "...", "text": "..."}'
   ```
