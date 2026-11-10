#1.1
import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

nltk.download('stopwords')
nltk.download('wordnet')

with open('/Users/jeremypharell/Desktop/£/School/Honours/ITNPA4-B34/Assignment/assignment resources/research_abstracts.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df.dropna(subset=['Abstract'])

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_abstract'] = df['Abstract'].apply(clean_text)
print(df[['Abstract', 'clean_abstract']].head())

#1.3
X = df['clean_abstract']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_train_pred = model.predict(X_train_vec)
y_test_pred = model.predict(X_test_vec)

print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred))
print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_test_pred))

cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5)
print("\nCross validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())
