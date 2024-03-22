# advanced_text_processing.py
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import random

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

def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]
