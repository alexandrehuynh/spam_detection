import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def preprocess_text(message):
    # Convert to lowercase
    message = message.lower()
    # Remove punctuation
    message = message.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(message)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def load_and_clean_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath, encoding='latin-1')
    # Adjusted to match the correct column names
    data.columns = ['label', 'message']  # Ensure this matches your CSV file's header
    # Clean the data
    data['message'] = data['message'].apply(preprocess_text)
    return data

