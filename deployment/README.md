# Researchly Abstract Classifier - Deployment

## Setup
1. Install dependencies: pip install -r requirements.txt
2. Ensure svm_model.pkl and tfidf_vectorizer.pkl are in this folder (run train_and_save.py if missing)
3. Start the app: python app.py
4. Open http://127.0.0.1:5000 in a browser

## Usage
Paste a research abstract into the textarea and click Classify Abstract. The prediction will appear below.

## Notes
The classifier uses a Linear SVM trained on TF-IDF features with balanced class weights. The input text is cleaned with the same pipeline used during training so that predictions are consistent.
