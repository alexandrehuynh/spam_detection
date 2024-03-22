import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import random
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def preprocess_text(message):
    # Convert to lowercase, remove punctuation, and tokenize
    message = message.lower()
    message = message.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(message)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Stemming/Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

def synonym_replacement(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonyms.add(synonym)
    if synonyms:
        return random.choice(list(synonyms))
    else:
        return word

def load_and_clean_data(filepath):
    data = pd.read_csv(filepath, encoding='latin-1')
    data.columns = ['label', 'message']
    data['message'] = data['message'].apply(preprocess_text)
    return data
