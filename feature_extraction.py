from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['message'])
    y = data['label']
    return X, y
