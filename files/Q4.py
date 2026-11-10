import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import numpy as np
import pyLDAvis
import pyLDAvis.lda_model # Fixed import for newer versions
import warnings

warnings.filterwarnings('ignore')

# --- Load and Clean Data (Same as Q3) ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

folder_path = '/Users/jeremypharell/Desktop/£/School/Honours/ITNPA4-B34/Assignment/assignment resources/research_corpus'
data = []
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            data.append({'article_id': filename, 'content': f.read()})

df_corpus = pd.DataFrame(data)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_corpus(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

df_corpus['clean_content'] = df_corpus['content'].apply(clean_corpus)

# --- Q4.1 LDA Topic Modeling ---
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
tf = tf_vectorizer.fit_transform(df_corpus['clean_content'])
tf_feature_names = tf_vectorizer.get_feature_names_out()

lda = LatentDirichletAllocation(n_components=3, max_iter=50, random_state=42)
lda.fit(tf)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

print("=== LDA Topics ===")
display_topics(lda, tf_feature_names, 10)

# --- Q4.2 pyLDAvis Interactive Map ---
print("\nGenerating pyLDAvis HTML file...")

# Prepare the visualization using the updated lda_model module
panel = pyLDAvis.lda_model.prepare(lda, tf, tf_vectorizer, mds='tsne')

# Save to an HTML file
pyLDAvis.save_html(panel, 'lda_visualization.html')
print("Visualization saved as lda_visualization.html. Open this file in your browser.")
