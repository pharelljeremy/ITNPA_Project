import pandas as pd
import json
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

nltk.download('stopwords')
nltk.download('wordnet')

with open('/Users/jeremypharell/Desktop/£/School/Honours/ITNPA4-B34/Assignment/assignment resources/research_abstracts.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data).dropna(subset=['Abstract'])
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

df['clean_abstract'] = df['Abstract'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(df['clean_abstract'])
model = LinearSVC(class_weight='balanced', random_state=42, max_iter=5000)
model.fit(X_vec, df['Label'])

with open('svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved.")
