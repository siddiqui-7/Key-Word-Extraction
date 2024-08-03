import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the pre-trained models and feature names
cv = pickle.load(open('count_vector.pkl', 'rb'))
tfidf_transformer = pickle.load(open('tfidf_transformer.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))

# Stop words
stop_words = set(stopwords.words('english'))
new_stop_words = ["fig", "figure", "image", "sample", "using", 
                  "show", "result", "large", "also", "one", 
                  "two", "three", "four", "five", "seven", 
                  "eight", "nine"]
stop_words = list(set(stop_words).union(set(new_stop_words)))

# Preprocessing function
def preprocess_text(txt):
    txt = txt.lower()
    txt = re.sub(r"<.*?>", " ", txt)
    txt = re.sub(r"[^a-zA-Z]", " ", txt)
    txt = nltk.word_tokenize(txt)
    txt = [word for word in txt if word not in stop_words]
    txt = [word for word in txt if len(word) >= 3]
    lmtr = WordNetLemmatizer()
    txt = [lmtr.lemmatize(word) for word in txt]
    return " ".join(txt)

# Function to sort coo_matrix
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

# Function to extract top n features
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    results = {feature_vals[idx]: score_vals[idx] for idx in range(len(feature_vals))}
    return results

# Function to get keywords
def get_keywords(doc):
    tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
    return keywords

# Load the dataset
df = pd.read_csv('papers.csv')

# Streamlit App
st.title('Paper Keyword Extractor')

paper_idx = st.number_input('Enter the paper index', min_value=0, max_value=len(df)-1, value=0)

if st.button('Extract Keywords'):
    paper_text = df['paper_text'][paper_idx]
    processed_text = preprocess_text(paper_text)
    keywords = get_keywords(processed_text)
    st.write(f"Title: {df['title'][paper_idx]}")
    st.write(f"Abstract: {df['abstract'][paper_idx]}")
    st.write("Keywords:")
    for k, v in keywords.items():
        st.write(f"{k}: {v}")

# Optionally display the DataFrame
if st.checkbox('Show DataFrame'):
    st.write(df.head())
