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
ham_messages = [
    "Hey, are you coming to the party tonight?", 
    "Can we meet tomorrow for lunch?", 
    "Just finished a great book on data science. Highly recommend it!",
    "Are we still on for the hiking trip this weekend?",
    "Happy Birthday! Hope you have a fantastic day filled with joy.",
    "The meeting has been rescheduled to next Thursday at 3 PM.",
    "Could you please send me the report by the end of the day?",
    "Looking forward to our dinner reservation at that new Italian restaurant.",
    "Reminder: Your doctor's appointment is tomorrow at 10:30 AM.",
    "I've attached the photos from our last trip. Let me know what you think!",
    "How's the project going? Need any help from my side?",
    "Good morning! Don't forget it's your turn to bring snacks for the team meeting."]
spam_messages = [
    "Claim your FREE trial of our new weight loss supplement now!",
    "You're selected! Take our exclusive survey to win a brand new iPad.",
    "Hot singles in your area waiting to meet you! Click now!",
    "You've won a cruise to the Bahamas! Call now to claim your ticket.",
    "Act now to extend your car's warranty before it's too late.",
    "Congratulations, you've been chosen for a limited-time offer to get 50% off our best-selling product.",
    "You owe $500 in taxes. Pay immediately through this link to avoid legal action.",
    "Exclusive deal just for you! Buy one get one free on all our products, today only.",
    "Your computer is infected with a virus! Download our antivirus software now to protect it.",
    "This is not a scam! You've won a $1000 gift card. Enter your details to claim."
]

# Generating dataset with augmented messages
augmentation_functions = [
    lambda msg: synonym_replacement(random.choice(msg.split())),  # No change needed here
    lambda msg: random_insertion(msg, n=2),  # Preset n=2 for example
    lambda msg: random_swap(msg, n=1),  # Preset n=1, default
    lambda msg: random_deletion(msg, p=0.1)  # Preset p=0.1, default
]

# Generating dataset with augmented messages
data = []
for _ in range(200):  # Increase the number for more data
    msg = random.choice(ham_messages)
    augmented_msg = random.choice(augmentation_functions)(msg)
    data.append(["ham", augmented_msg])

    msg = random.choice(spam_messages)
    augmented_msg = random.choice(augmentation_functions)(msg)
    data.append(["spam", augmented_msg])

# Save to CSV
df = pd.DataFrame(data, columns=['label', 'message'])
df.to_csv('spam.csv', index=False)
