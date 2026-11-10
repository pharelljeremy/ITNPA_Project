from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)

with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    abstract_text = ''
    if request.method == 'POST':
        abstract_text = request.form.get('abstract', '')
        if abstract_text.strip():
            cleaned = clean_text(abstract_text)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]
    return render_template('index.html', prediction=prediction, abstract=abstract_text)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
