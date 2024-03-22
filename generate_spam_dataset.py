import pandas as pd
import random
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')

# Function to replace a word with one of its synonyms
def synonym_replacement(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " ").replace("-", " ").lower())
    if synonyms:
        return random.choice(list(synonyms))
    else:
        return word

# Example augmentation function that replaces random words with synonyms
def augment_message_by_synonym_replacement(message):
    words = message.split()
    for i in range(len(words)):
        if random.random() < 0.2:  # 20% chance to replace each word with a synonym
            words[i] = synonym_replacement(words[i])
    return ' '.join(words)

# Original spam and ham messages
ham_messages = ["Hey, are you coming to the party tonight?", "Can we meet tomorrow for lunch?"]
spam_messages = ["Congratulations! You've won a $1000 Walmart gift card. Go to our site to claim now.", "You have been selected for a chance to win an iPhone. Click here to claim your prize."]

# Generating dataset with augmented messages
data = []
for _ in range(100):  # Increase or decrease the range for more or less data
    message = random.choice(ham_messages)
    augmented_message = augment_message_by_synonym_replacement(message)
    data.append(["ham", augmented_message])

    message = random.choice(spam_messages)
    augmented_message = augment_message_by_synonym_replacement(message)
    data.append(["spam", augmented_message])

# Creating a DataFrame and saving to CSV
df = pd.DataFrame(data, columns=['label', 'message'])
df.to_csv('spam.csv', index=False)
