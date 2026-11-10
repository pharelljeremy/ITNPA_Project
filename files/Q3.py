#3.1

import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set the path to your research_corpus folder
folder_path = '/Users/jeremypharell/Desktop/£/School/Honours/ITNPA4-B34/Assignment/assignment resources/research_corpus'

data = []
# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            data.append({'article_id': filename, 'content': content})

# Create the dataframe
df_corpus = pd.DataFrame(data)

print("Dataframe shape:", df_corpus.shape)
print("\nFirst few rows:")
print(df_corpus.head())

#3.2
nltk.download('stopwords')
nltk.download('wordnet')

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
print(df_corpus[['article_id', 'clean_content']].head())

#3.3
nltk.download('punkt')

first_article_text = df_corpus['content'].iloc[0]
first_article_clean = df_corpus['clean_content'].iloc[0]

# Summarization
parser = PlaintextParser.from_string(first_article_text, Tokenizer("english"))
summarizer = TextRankSummarizer()
summary = summarizer(parser.document, 4) # Extract top 4 sentences

print("=== Summarized First Article ===")
for sentence in summary:
    print(sentence)

# Visualization
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(first_article_clean)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for First Article (research_4.txt)')
plt.show()

#3.4
# --- Q3.4 Code ---
from sumy.summarizers.lsa import LsaSummarizer
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Aggregate the remaining articles
remaining_text = " ".join(df_corpus['content'].iloc[1:].tolist())
remaining_clean = " ".join(df_corpus['clean_content'].iloc[1:].tolist())

# Summarization using LSA (Latent Semantic Analysis)
parser_rem = PlaintextParser.from_string(remaining_text, Tokenizer("english"))
lsa_summarizer = LsaSummarizer()
lsa_summary = lsa_summarizer(parser_rem.document, 5) # Extract top 5 sentences

print("=== Aggregated Summary of Remaining Articles (LSA) ===")
for sentence in lsa_summary:
    print(sentence)

# Visualization: Top 15 most common words in the remaining corpus
words = remaining_clean.split()
word_counts = Counter(words).most_common(15)
words_df = pd.DataFrame(word_counts, columns=['Word', 'Frequency'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency', y='Word', data=words_df, palette='viridis')
plt.title('Top 15 Most Frequent Words in Remaining Articles')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()
