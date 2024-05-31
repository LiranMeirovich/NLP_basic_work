import nltk
import spacy
import pandas as pd
from collections import Counter
import re
from spacy.lang.en import English
# nltk.download("punkt")
# nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import random
from spacy.tokenizer import Tokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

df = pd.read_csv("./spam.csv", encoding='latin-1')
print("______PART 1 - Data Loading & Basic Analysis______\n")

print("     The total number of SMS messages  in spam is: ", df['v2'].count(),"\n")

print("     The Number of ham's is: ", df[df['v1'] == 'ham'].shape[0] ," The Number of spam's is: ", df[df['v1'] == 'spam'].shape[0],"\n")

df['word_count'] = df['v2'].apply(lambda x: len(str(x).split()))
average_word_count = df['word_count'].mean()
print("     The average number of words per message is: ", average_word_count,"\n")
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

print("     The 5 most frequent words are:","\n")
for word, count in most_common_words:
    print(f"     {word}: {count}")

single_words = 0
for word,count in word_counts.items():
    if count == 1:
       single_words+=1
print("\n     There are ",single_words ," words that appear once\n")

print("______PART 2 - Text Processing______\n")
corpus = []
with open('spam.csv', 'r') as f:
    for line in f:
        corpus.append(" ".join(line.split()[1:]))

# Preprocess the data
def nltk_preprocess(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stopwords and token.isalpha()]
    # Calculate word frequencies
    # frequency_dist = nltk.FreqDist(filtered_tokens)

    return filtered_tokens


# nltk_tokenized_set = []
# for line in corpus:
#     nltk_tokenized_set.append((nltk_preprocess(line)))

nlp = spacy.load("en_core_web_sm")
def spacy_preprocess(string):
    # Process the document using spaCy
    doc = nlp(string)
    # Remove stopwords and non-alphabetic tokens, and convert to lowercase
    filtered_tokens = [token for token in doc if not token.is_stop and token.is_alpha]
    return filtered_tokens

spacy_tokenized_set = [((spacy_preprocess(line))) for line in corpus]

# print(spacy_tokenized_set)

# spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")


def spacy_lemmatize(tokens):
    # Process the document using spaCy
    # print(tokens)
    lemmatized_tokens = [token.lemma_ for token in tokens]
    # print(lemmatized_tokens)

    return [lemmatized_tokens]

spacy_lemmatized_tokens = [spacy_lemmatize(tokens) for tokens in spacy_tokenized_set]

# print(spacy_lemmatized_tokens)

#
# print(nltk_tokenized_set[2])
print(spacy_lemmatized_tokens[2])
print(spacy_tokenized_set[2])



