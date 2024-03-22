import pandas as pd
import random

# Sample messages
ham_messages = [
    "Hey, are you coming to the party tonight?",
    "Can we meet tomorrow for lunch?",
    "Don't forget to submit your assignment by tomorrow.",
    "Got your email, I'll review and get back to you by Monday.",
    "Let's catch up over coffee next week?"
]

spam_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to our site to claim now.",
    "You have been selected for a chance to win an iPhone. Click here to claim your prize.",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)",
    "Urgent! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 & claim your prize."
]

# Generating dataset
data = []
for _ in range(50):  # Generate 50 ham messages
    data.append(["ham", random.choice(ham_messages)])
for _ in range(50):  # Generate 50 spam messages
    data.append(["spam", random.choice(spam_messages)])

# Shuffle the dataset
random.shuffle(data)

# Creating a DataFrame and saving to CSV
df = pd.DataFrame(data, columns=['label', 'message'])
df.to_csv('spam.csv', index=False)
