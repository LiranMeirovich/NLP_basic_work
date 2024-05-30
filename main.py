import nltk
import spacy
import pandas as pd
from collections import Counter
import re

df = pd.read_csv("./spam.csv", encoding='latin-1')
print("______PART 1 - Data Loading & Basic Analysis______")

print("     The total number of SMS messages  in spam is: ", df['v2'].count())

print("     The Number of ham's is: ", df[df['v1'] == 'ham'].shape[0] ," The Number of spam's is: ", df[df['v1'] == 'spam'].shape[0])

df['word_count'] = df['v2'].apply(lambda x: len(str(x).split()))

average_word_count = df['word_count'].mean()

print("     The average number of words per message is: ", average_word_count)

# Function to tokenize the messages
def tokenize(text):
    # Convert to lower case and use regex to find words
    words = re.findall(r'\b\w+\b', text.lower())
    return words

# Apply the tokenization function to the messages and flatten the list of lists
all_words = df['v2'].apply(tokenize).explode()

# Count the frequency of each word
word_counts = Counter(all_words)

# Get the 5 most common words
most_common_words = word_counts.most_common(5)

print("     The 5 most frequent words are:")
for word, count in most_common_words:
    print(f"     {word}: {count}")

single_words = 0
for word,count in word_counts.items():
    if count == 1:
        single_words+=1
print("     There are ",single_words ," words that appear once")
