#2.1

import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score

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

X = df['clean_abstract']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
    'Linear SVM': LinearSVC(class_weight='balanced', random_state=42, max_iter=5000),
    'Multinomial NB': MultinomialNB(),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    cv = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='f1_macro')
    results[name] = {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'cv_mean': cv.mean(),
        'cv_std': cv.std()
    }
    print(f"=== {name} ===")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Macro F1 (Test): {macro_f1:.4f}")
    print(f"CV Macro F1: {cv.mean():.4f} (+/- {cv.std():.4f})")
    print(classification_report(y_test, y_pred))
    print()

print("Summary:")
for name, r in results.items():
    print(f"{name}: Acc={r['accuracy']:.4f}, MacroF1={r['macro_f1']:.4f}, CV={r['cv_mean']:.4f} (+/- {r['cv_std']:.4f})")

#2.2
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
X = df['clean_abstract']
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Baseline unigram SVM (from Q2.1)
uni_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
X_train_uni = uni_vec.fit_transform(X_train)
X_test_uni = uni_vec.transform(X_test)
svm_uni = LinearSVC(class_weight='balanced', random_state=42, max_iter=5000)
svm_uni.fit(X_train_uni, y_train)
y_pred_uni = svm_uni.predict(X_test_uni)

# Enhanced n-gram SVM
ngram_vec = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), min_df=2)
X_train_ng = ngram_vec.fit_transform(X_train)
X_test_ng = ngram_vec.transform(X_test)
svm_ng = LinearSVC(class_weight='balanced', random_state=42, max_iter=5000)
svm_ng.fit(X_train_ng, y_train)
y_pred_ng = svm_ng.predict(X_test_ng)

for label, yp, model, Xtr in [
    ('Unigram SVM (Q2.1)', y_pred_uni, svm_uni, X_train_uni),
    ('N-gram SVM (Q2.2)', y_pred_ng, svm_ng, X_train_ng),
]:
    acc = accuracy_score(y_test, yp)
    mf1 = f1_score(y_test, yp, average='macro')
    cv = cross_val_score(model, Xtr, y_train, cv=5, scoring='f1_macro')
    print(f"=== {label} ===")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Macro F1: {mf1:.4f}")
    print(f"CV Macro F1: {cv.mean():.4f} (+/- {cv.std():.4f})")
    print(classification_report(y_test, yp))
