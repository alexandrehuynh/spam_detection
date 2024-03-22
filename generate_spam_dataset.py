import pandas as pd
import random
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

# Synonym replacement
def synonym_replacement(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " ").replace("-", " ").lower())
    if synonyms:
        return random.choice(list(synonyms))
    else:
        return word

# Random insertion
def random_insertion(sentence, n):
    words = sentence.split()
    for _ in range(n):
        new_word = synonym_replacement(random.choice(words))
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, new_word)
    return ' '.join(words)

# Random swap
def random_swap(sentence, n=1):
    words = sentence.split()
    for _ in range(n):
        pos1, pos2 = random.sample(range(len(words)), 2)
        words[pos1], words[pos2] = words[pos2], words[pos1]
    return ' '.join(words)

# Random deletion
def random_deletion(sentence, p=0.1):
    words = sentence.split()
    if len(words) == 1:  # return if single word
        return sentence
    remaining = list(filter(lambda x: random.random() > p, words))  # randomly delete words
    if len(remaining) == 0:  # if all words are deleted, pick a random word
        return random.choice(words)
    return ' '.join(remaining)

# Example original messages
ham_messages = ["Hey, are you coming to the party tonight?", "Can we meet tomorrow for lunch?"]
spam_messages = ["Congratulations! You've won a $1000 Walmart gift card. Go to our site to claim now.", "You have been selected for a chance to win an iPhone. Click here to claim your prize."]

# Generating dataset with augmented messages
data = []
for _ in range(200):  # Increase the number for more data
    msg = random.choice(ham_messages)
    augmented_msg = random.choice([synonym_replacement, random_insertion, random_swap, random_deletion])(msg)
    data.append(["ham", augmented_msg])

    msg = random.choice(spam_messages)
    augmented_msg = random.choice([synonym_replacement, random_insertion, random_swap, random_deletion])(msg)
    data.append(["spam", augmented_msg])

# Save to CSV
df = pd.DataFrame(data, columns=['label', 'message'])
df.to_csv('spam.csv', index=False)
