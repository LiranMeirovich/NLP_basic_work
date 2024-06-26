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

print("______PART 5.a - BoW______\n")
words = []
for i in lemmatized_tokens:
    words.append(' '.join(i))


def bow(corpus):
    vectorizer = CountVectorizer()

    # Fit the vectorizer on the corpus and transform the corpus into a BOW matrix
    return vectorizer.fit_transform(corpus).toarray(), vectorizer.get_feature_names_out()

bow_matrix, feature_names= bow(words)

# Print the BOW matrix
print(bow_matrix)

# Print the feature names (i.e., unique words) learned by the vectorizer
print(feature_names)

print("______PART 5.b - TF-IDF______\n")

from sklearn.feature_extraction.text import TfidfVectorizer

# Create and apply TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(words)

print(tfidf_matrix)

print("______PART 5.c - Word2Vec______\n")


# Import necessary libraries
from gensim.models import Word2Vec
from nltk.corpus import brown

# Train the Word2Vec model
model = Word2Vec(
    sentences=lemmatized_tokens,      # The corpus to train the model on
    vector_size=100,       # The size of the word vectors to be learned
    window=5,              # The size of the window of words to be considered
    min_count=1,           # The minimum frequency required for a word to be included in the vocabulary
    sg=0,                  # 0 for CBOW, 1 for skip-gram
    negative=5,            # The number of negative samples to use for negative sampling
    ns_exponent=0.75,      # The exponent used to shape the negative sampling distribution
    alpha=0.03,            # The initial learning rate
    min_alpha=0.0007,      # The minimum learning rate to which the learning rate will be linearly reduced
    epochs=30,             # The number of epochs (iterations) over the corpus
    workers=4,             # The number of worker threads to use for training the model
    seed=42,               # The seed for the random number generator
    max_vocab_size=None    # The maximum vocabulary size (None means no limit)
)

# Get the vector representation of a word

vector = model.wv['three']

# Find the most similar words to a given word
similar_words = model.wv.most_similar('three')

# Print the vector and similar words
print("Vector for 'three':", vector)
print("Most similar words to 'three':", similar_words)

print("______PART 6 - GLOVE______\n")

import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

#The glove algorithm we were shown was applied on a list of words
#And so we put all the words in the lemmatized list into wordsArr
wordsArr = []

for line in lemmatized_tokens:
    for word in line:
        wordsArr.append(word)

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


X, word_to_id = build_cooccurrence_matrix(wordsArr)
word_vectors = train_glove(X, vector_size=5, iterations=100)

# Print word vectors, only printing 5 because that it's a lot of words
for i, (word, idx) in enumerate(word_to_id.items()):
    if i >= 5:
        break
    print(f"{word}: {word_vectors[idx]}")


print("______PART 7 - CYK______\n")
def cyk_parse(sentence, grammar):
    # Step 1: Tokenization
    tokens = sentence.split()
    n = len(tokens)
    table = [[set() for _ in range(n+1)] for _ in range(n+1)]

    # Step 2: Initialization
    for i in range(1, n+1):
        for rule in grammar:
            if rule[1] == tokens[i-1]:
                table[i][i].add(rule[0])

    # Step 3: Rule Application
    for length in range(2, n+1):
        for i in range(1, n-length+2):
            j = i + length - 1
            for k in range(i, j):
                for rule in grammar:
                    if len(rule) == 3:
                        for left in table[i][k]:
                            for right in table[k+1][j]:
                                if rule[1] in left and rule[2] in right:
                                    table[i][j].add(rule[0])

    # Step 4: Backtracking
    if 'S' in table[1][n]:
        return True, table
    else:
        return False, table


# Example usage:

# Define the context-free grammar in CNF
# There are algorithms that do this automatically
# But for time management we decided to hard-code them using Claude
grammars = [
    # Grammar for "brain many tabs open"
    [
        ('S', 'NP', 'Adj'),
        ('NP', 'Noun', 'NP'),
        ('NP', 'Adj', 'Noun'),
        ('Noun', 'brain'),
        ('Noun', 'tabs'),
        ('Adj', 'many'),
        ('Adj', 'open')
    ],

    # Grammar for "reading book impossible put"
    [
        ('S', 'VP', 'Verb'),
        ('VP', 'VP', 'Adj'),
        ('VP', 'Verb', 'Noun'),
        ('Verb', 'reading'),
        ('Verb', 'put'),
        ('Noun', 'book'),
        ('Adj', 'impossible')
    ],

    # Grammar for "used think indecisive sure"
    [
        ('S', 'VP', 'AdjP'),
        ('VP', 'Verb', 'Verb'),
        ('AdjP', 'Adj', 'Adj'),
        ('Verb', 'used'),
        ('Verb', 'think'),
        ('Adj', 'indecisive'),
        ('Adj', 'sure')
    ],

    # Grammar for "fall floor needed hug"
    [
        ('S', 'VP', 'NP'),
        ('VP', 'Verb', 'Noun'),
        ('NP', 'Adj', 'Noun'),
        ('Verb', 'fall'),
        ('Noun', 'floor'),
        ('Adj', 'needed'),
        ('Noun', 'hug')
    ],

    # Grammar for "roll Butter call biscuit"
    [
        ('S', 'VP', 'NP'),
        ('VP', 'Verb', 'Noun'),
        ('NP', 'Verb', 'Noun'),
        ('Verb', 'roll'),
        ('Verb', 'call'),
        ('Noun', 'Butter'),
        ('Noun', 'biscuit')
    ]
]

# Algorithm works on whole sentences and so we will make a list of strings where
# each string is a sentence.
sentences = []
for i in tokenized_set:
    sentences.append(' '.join(i))

# We decided to take 5 sentences that are of length 4 to make it not too difficult to hardcode
# them but also to show a table which isn't just 2 dimensional and boring
sentences_with_four_words = [sentence for sentence in sentences if len(sentence.split()) == 4]
selected_sentences = sentences_with_four_words[:5]

print(selected_sentences)

# Input sentence to be parsed
# sentence = "the cat chased a dog"

# Call the CYK parser
for i in range(0, len(selected_sentences)):
    parsed, table = cyk_parse(selected_sentences[i], grammars[i])

    if parsed:
        print("Input sentence: ", selected_sentences[i])
        print("Parse table: ")
        for row in table:
            print(row)
    else:
        print("Input sentence: ", selected_sentences[i])
        print("Sentence not parsed.")

# Print the parse table and whether the sentence was parsed or not
