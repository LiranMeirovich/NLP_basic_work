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

print("     The total number of SMS messages  in spam is: ", df['v2'].count(), "\n")

print("     The Number of ham's is: ", df[df['v1'] == 'ham'].shape[0], " The Number of spam's is: ",
      df[df['v1'] == 'spam'].shape[0], "\n")

df['word_count'] = df['v2'].apply(lambda x: len(str(x).split()))
average_word_count = df['word_count'].mean()
print("     The average number of words per message is: ", average_word_count, "\n")


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

print("     The 5 most frequent words are:", "\n")
for word, count in most_common_words:
    print(f"     {word}: {count}")

single_words = 0
for word, count in word_counts.items():
    if count == 1:
        single_words += 1
print("\n     There are ", single_words, " words that appear once\n")

print("______PART 2 - Text Processing______\n")

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict
# nltk.download('averaged_perceptron_tagger')
#
#
#
# import time
#
# corpus = []
# with open('spam.csv', 'r') as f:
#     for line in f:
#         corpus.append(" ".join(line.split()[1:]))
#
# # Preprocess the data
def nltk_preprocess(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords and token.isalpha()]
    # Calculate word frequencies
    # frequency_dist = nltk.FreqDist(filtered_tokens)

    return filtered_tokens

tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

lemma_function = WordNetLemmatizer()

def nltk_lemmatize(tokens):
    lemmatized_tokens = []
    for token, tag in pos_tag(tokens):
        lemmatized_tokens.append(lemma_function.lemmatize(token, tag_map[tag[0]]))

    return lemmatized_tokens

# # nltk.download('wordnet')
#
# start_time = time.time()
# nltk_tokenized_set = []
# for line in corpus:
#     nltk_tokenized_set.append((nltk_preprocess(line)))
# end_time = time.time()
# print(f"NLTK Tokenization Time: {end_time - start_time:.6f} seconds")
#
# start_time = time.time()
# nltk_lemmatized_set = [nltk_lemmatize(tokens) for tokens in nltk_tokenized_set]
# end_time = time.time()
# print(f"NLTK Lemmatization Time: {end_time - start_time:.6f} seconds")
#
def stem_tokens(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens
#
# start_time = time.time()
# nltk_stemmed_set = [stem_tokens(tokens) for tokens in nltk_tokenized_set]
# end_time = time.time()
# print(f"NLTK Stemming Time: {end_time - start_time:.6f} seconds")
#
# nlp = spacy.load("en_core_web_sm")
# def spacy_preprocess(string):
#     # Process the document using spaCy
#     doc = nlp(string)
#     # Remove stopwords and non-alphabetic tokens, and convert to lowercase
#     filtered_tokens = [token for token in doc if not token.is_stop and token.is_alpha]
#     return filtered_tokens
#
# start_time = time.time()
# spacy_tokenized_set = [((spacy_preprocess(line))) for line in corpus]
# end_time = time.time()
# print(f"spaCy Tokenization Time: {end_time - start_time:.6f} seconds")
#
# # spacy.cli.download("en_core_web_sm")
#
# nlp = spacy.load("en_core_web_sm")
#
# def spacy_lemmatize(tokens):
#     # Process the document using spaCy
#     # print(tokens)
#     lemmatized_tokens = [token.lemma_ for token in tokens]
#     # print(lemmatized_tokens)
#
#     return [lemmatized_tokens]
#
# start_time = time.time()
# spacy_lemmatized_tokens = [spacy_lemmatize(tokens) for tokens in spacy_tokenized_set]
# end_time = time.time()
# print(f"spaCy Lemmatization Time: {end_time - start_time:.6f} seconds")
#
# def stem_tokens_spacy(tokens):
#     stemmer = PorterStemmer()
#     #spacy tokens are held as objects, they are being numbered and to access
#     #the text you need to access it trough the object.
#     stemmed_tokens = [stemmer.stem(token.text) for token in tokens]
#     return stemmed_tokens
#
# start_time = time.time()
# spacy_stemmed_tokens = [stem_tokens_spacy(tokens) for tokens in spacy_tokenized_set]
# end_time = time.time()
# print(f"spaCy Stemming Time: {end_time - start_time:.6f} seconds")
#
# print()
# print("Example 95:")
# print("NLTK Tokenized Set:", nltk_tokenized_set[95])
# print("NLTK Lemmatized Set:", nltk_lemmatized_set[95])
# print("NLTK Stemmed Set:", nltk_stemmed_set[95])
# print("spaCy Tokenized Set:", spacy_tokenized_set[95])
# print("spaCy Lemmatized Tokens:", spacy_lemmatized_tokens[95])
# print("spaCy Stemmed Tokens:", spacy_stemmed_tokens[95])
# print()
# print("Example 686:")
# print("NLTK Tokenized Set:", nltk_tokenized_set[686])
# print("NLTK Lemmatized Set:", nltk_lemmatized_set[686])
# print("NLTK Stemmed Set:", nltk_stemmed_set[686])
# print("spaCy Tokenized Set:", spacy_tokenized_set[686])
# print("spaCy Lemmatized Tokens:", spacy_lemmatized_tokens[686])
# print("spaCy Stemmed Tokens:", spacy_stemmed_tokens[686])
#
# #For output format, we can see that for example nltk removed I since it considers it
# #a stopword, additionally there will be differences in words like can't
# #one will take it as can + 't and the other as ca + n't
# #For proccessing speed we can see that spacy takes MUCH longer than nltk
# #This is because there are a lot more things happening in spaCy, for example
# #lemmatization happens almost instantly on spaCy because it adds it as an attribute
# #during tokenization
# #Another interesting thing we can see is that lemmatization with spaCy is automatic
# #meanwhile for nltk we need to add a map which describes for nltk which
# #word is verb, adjective etc.
#
# # Print word count and top 5 frequent words for NLTK tokens
# all_nltk_tokens = [token for tokens in nltk_tokenized_set for token in tokens]
# word_count = len(all_nltk_tokens)
# word_freq = Counter(all_nltk_tokens)
# top_5_words = word_freq.most_common(5)
# print()
# print("Statistics for nltk:")
# print(f"NLTK Word Count: {word_count}")
# print(f"NLTK Top 5 Frequent Words: {top_5_words}")
#
# # Print word count and top 5 frequent words for spaCy tokens
# all_spacy_tokens = [token.text for tokens in spacy_tokenized_set for token in tokens]
# word_count = len(all_spacy_tokens)
# word_freq = Counter(all_spacy_tokens)
# top_5_words = word_freq.most_common(5)
# print()
# print("Statistics for spaCy:")
# print(f"spaCy Word Count: {word_count}")
# print(f"spaCy Top 5 Frequent Words: {top_5_words}")


print("______PART 3 - WEB SCRAPING______\n")

from bs4 import BeautifulSoup
from requests import get
from fake_user_agent import user_agent

ua = user_agent("chrome")


def lovely_soup(u):
    r = get(u, headers={'User-Agent': ua})
    return BeautifulSoup(r.text, 'lxml')


url = 'https://old.reddit.com/r/todayilearned/top/'
soup = lovely_soup(url)

titles = soup.findAll('p', {'class': 'title'})


nltk_tokenized_titles = []
for line in titles:
    nltk_tokenized_titles.append((nltk_preprocess(line.text)))

nltk_lemmatized_titles = [nltk_lemmatize(tokens) for tokens in nltk_tokenized_titles]
nltk_stemmed_titles = [stem_tokens(tokens) for tokens in nltk_tokenized_titles]

# Print the first tokenized title
print("Tokenized Title:", nltk_tokenized_titles[0])

# Print the first lemmatized title
print("Lemmatized Title:", nltk_lemmatized_titles[0])

# Print the first stemmed title
print("Stemmed Title:", nltk_stemmed_titles[0])

# Extract text from titles and make them lowercase
original_texts = [line.text.lower() for line in titles]

# Count word frequencies before tokenization
original_word_count = Counter()
for text in original_texts:
    original_word_count.update(text.split())

# Count word frequencies after tokenization
tokenized_word_count = Counter()
for tokens in nltk_tokenized_titles:
    tokenized_word_count.update(tokens)

# Get the top 5 most common words
top_5_original = original_word_count.most_common(5)
top_5_tokenized = tokenized_word_count.most_common(5)

# Calculate total word counts
total_words_original = sum(original_word_count.values())
total_words_tokenized = sum(tokenized_word_count.values())

# Print results
print()
print("Before Tokenization:")
print("Top 5 Most Frequently Used Words:", top_5_original)
print("Total Word Count:", total_words_original)

print("\nAfter Tokenization:")
print("Top 5 Most Frequently Used Words:", top_5_tokenized)
print("Total Word Count:", total_words_tokenized)