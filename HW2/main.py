import nltk
import nltk.data

import spacy
import pandas as pd
from collections import Counter
import re
from spacy.lang.en import English

# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from spacy.lang.he import Hebrew
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

# Preprocess the data
def nltk_preprocess(text, language):
    tokens = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words(language)
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords and token.isalpha()]
    return filtered_tokens


tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

lemma_function = WordNetLemmatizer()


def nltk_lemmatize(tokens):
    lemmatized_tokens = []
    for token, tag in pos_tag(tokens):
        lemmatized_tokens.append(lemma_function.lemmatize(token))
    return lemmatized_tokens


print("______PART 1-4 - Preperation of data______\n")


timestamp_pattern = r'\[\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2}\]'

with open('chat.txt', 'r', encoding='utf-8') as file:
    whatsApp_corpus = []
    for line in file:
        cleaned_line = re.sub(timestamp_pattern, '', line).strip()
        cleaned_line = cleaned_line.split(':', 1)[-1].strip()
        whatsApp_corpus.append(cleaned_line)


tokenized_set = [nltk_preprocess(line, "english") for line in whatsApp_corpus]
lemmatized_tokens = [nltk_lemmatize(tokens) for tokens in tokenized_set]

# for i in range(0, len(lemmatized_tokens)):
#     print(lemmatized_tokens[i])
#     print(tokenized_set[i])


print("Original message:", whatsApp_corpus[5])
print("Tokenized message:", tokenized_set[5])
print("Lemmatized message:", lemmatized_tokens[5])
#
# print("\nComparisons of word statistics before and after processing\n")
# print("The total number of SMS messages in chat.txt is: ", len(whatsApp_corpus))

# Import necessary libraries

# Define sample corpus of documents

# Create an instance of CountVectorizer with default settings

words = []
for i in lemmatized_tokens:
    words.append(' '.join(i))


def bow(corpus):
    vectorizer = CountVectorizer()

    # Fit the vectorizer on the corpus and transform the corpus into a BOW matrix
    return vectorizer.fit_transform(corpus).toarray(), vectorizer.get_feature_names_out()

# bow_matrix, feature_names= bow(words)
#
# # Print the BOW matrix
# print(bow_matrix)
#
# # Print the feature names (i.e., unique words) learned by the vectorizer
# print(feature_names)

from sklearn.feature_extraction.text import TfidfVectorizer

# # Create and apply TF-IDF vectorizer
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(words)
#
# print(tfidf_matrix)

# Import necessary libraries
from gensim.models import Word2Vec
from nltk.corpus import brown

# Train the Word2Vec model
# model = Word2Vec(
#     sentences=lemmatized_tokens,      # The corpus to train the model on
#     vector_size=100,       # The size of the word vectors to be learned
#     window=5,              # The size of the window of words to be considered
#     min_count=1,           # The minimum frequency required for a word to be included in the vocabulary
#     sg=0,                  # 0 for CBOW, 1 for skip-gram
#     negative=5,            # The number of negative samples to use for negative sampling
#     ns_exponent=0.75,      # The exponent used to shape the negative sampling distribution
#     alpha=0.03,            # The initial learning rate
#     min_alpha=0.0007,      # The minimum learning rate to which the learning rate will be linearly reduced
#     epochs=30,             # The number of epochs (iterations) over the corpus
#     workers=4,             # The number of worker threads to use for training the model
#     seed=42,               # The seed for the random number generator
#     max_vocab_size=None    # The maximum vocabulary size (None means no limit)
# )
#
# print(model)
#
# # Get the vector representation of a word
#
# vector = model.wv['three']
#
# # Find the most similar words to a given word
# similar_words = model.wv.most_similar('three')
#
# # Print the vector and similar words
# print("Vector for 'three':", vector)
# print("Most similar words to 'three':", similar_words)
#
import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

wordsArr = []

for fuck in lemmatized_tokens:
    for word in fuck:
        wordsArr.append(word)

print(type(wordsArr))


def build_cooccurrence_matrix(corpus, window_size=2):
    vocab = list(set(corpus))
    word_to_id = {word: i for i, word in enumerate(vocab)}
    cooccurrence = sparse.lil_matrix((len(vocab), len(vocab)), dtype=np.float64)

    for i, word in enumerate(corpus):
        left_context = max(0, i - window_size)
        right_context = min(len(corpus), i + window_size + 1)
        for j in range(left_context, right_context):
            if i != j:
                cooccurrence[word_to_id[word], word_to_id[corpus[j]]] += 1

    return cooccurrence.tocsr(), word_to_id


def glove_loss(X, W, b, U, c):
    return np.sum(np.power(W.dot(U.T) + b[:, np.newaxis] + c[np.newaxis, :] - np.log(X.toarray() + 1), 2))


def train_glove(X, vector_size=50, iterations=50, learning_rate=0.001):
    vocab_size = X.shape[0]
    W = np.random.randn(vocab_size, vector_size) / np.sqrt(vector_size)
    U = np.random.randn(vocab_size, vector_size) / np.sqrt(vector_size)
    b = np.zeros(vocab_size)
    c = np.zeros(vocab_size)

    for _ in range(iterations):
        grad_W = 2 * (W.dot(U.T) + b[:, np.newaxis] + c[np.newaxis, :] - np.log(X.toarray() + 1)).dot(U)
        grad_U = 2 * (W.dot(U.T) + b[:, np.newaxis] + c[np.newaxis, :] - np.log(X.toarray() + 1)).T.dot(W)
        grad_b = 2 * np.sum(W.dot(U.T) + b[:, np.newaxis] + c[np.newaxis, :] - np.log(X.toarray() + 1), axis=1)
        grad_c = 2 * np.sum(W.dot(U.T) + b[:, np.newaxis] + c[np.newaxis, :] - np.log(X.toarray() + 1), axis=0)

        W -= learning_rate * grad_W
        U -= learning_rate * grad_U
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c

        if _ % 10 == 0:
            print(f"Iteration {_}, Loss: {glove_loss(X, W, b, U, c)}")

    return (W + U) / 2


# Example usage
corpus = ["the", "cat", "sat", "on", "the", "mat", "the", "dog", "sat", "on", "the", "floor"]
X, word_to_id = build_cooccurrence_matrix(wordsArr)
word_vectors = train_glove(X, vector_size=5, iterations=100)

# Print word vectors
for word, idx in word_to_id.items():
    print(f"{word}: {word_vectors[idx]}")

