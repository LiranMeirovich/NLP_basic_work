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

# print("______PART 5.a - BoW______\n")
# words = []
# for i in lemmatized_tokens:
#     words.append(' '.join(i))
#
#
# def bow(corpus):
#     vectorizer = CountVectorizer()
#
#     # Fit the vectorizer on the corpus and transform the corpus into a BOW matrix
#     return vectorizer.fit_transform(corpus).toarray(), vectorizer.get_feature_names_out()
#
# bow_matrix, feature_names= bow(words)
#
# # Print the BOW matrix
# print(bow_matrix)
#
# # Print the feature names (i.e., unique words) learned by the vectorizer
# print(feature_names)
#
# print("______PART 5.b - TF-IDF______\n")
#
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# # Create and apply TF-IDF vectorizer
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(words)
#
# print(tfidf_matrix)

# print("______PART 5.c - Word2Vec______\n")
#
#
# # Import necessary libraries
# from gensim.models import Word2Vec
# from nltk.corpus import brown
#
# # Train the Word2Vec model
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
# # Get the vector representation of a word
#
# vector = model.wv['three']
#
# # Find the most similar words to a given word
# similar_words = model.wv.most_similar('three')

# # Print the vector and similar words
# print("Vector for 'three':", vector)
# print("Most similar words to 'three':", similar_words)

print("Part 2 RNN")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SimpleRNN, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Assume whatsApp_corpus is a list of strings, each string being a message or a line of text
# If it's not already in this format, you'll need to preprocess it accordingly
#
# Tokenize the text
test_texts = whatsApp_corpus[:3]  # Take the first 3 messages as test seeds
tokenizer = Tokenizer()
whatsApp_corpus = whatsApp_corpus[3:]
tokenizer.fit_on_texts(whatsApp_corpus)
total_words = len(tokenizer.word_index) + 1

# Create input sequences and output words
input_sequences = []
for line in whatsApp_corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create predictors and label
X, y = input_sequences[:,:-1], input_sequences[:,-1]

# Convert y to one-hot encoding
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Define and compile the RNN model
rnn_model = Sequential([
    Input(shape=(max_sequence_len-1,)),
    Embedding(total_words, 64),
    SimpleRNN(64, activation='tanh'),
    Dense(total_words, activation='softmax')
])

rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the RNN model
rnn_history = rnn_model.fit(X, y, epochs=50, validation_split=0.1, verbose=1)

# Function to generate text
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted, axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Test the RNN model with some seed texts from your corpus
for test_text in test_texts:
    seed = ' '.join(test_text.split()[:3])  # Use first 3 words of each message as seed
    print(f"Seed: {seed}")
    print(f"Generated: {generate_text(seed, 5, rnn_model, max_sequence_len)}")
    print()

print("Part 3 LSTM")

# Define and compile the LSTM model
lstm_model = Sequential([
    Input(shape=(max_sequence_len-1,)),
    Embedding(total_words, 100),  # Increased embedding dimension
    LSTM(150),  # Increased LSTM units
    Dense(total_words, activation='softmax')
])

lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the LSTM model
lstm_history = lstm_model.fit(X, y, epochs=50, validation_split=0.1, verbose=1)

# Test the LSTM model with some seed texts from your corpus
for test_text in test_texts:
    seed = ' '.join(test_text.split()[:3])  # Use first 3 words of each message as seed
    print(f"Seed: {seed}")
    print(f"Generated: {generate_text(seed, 5, lstm_model, max_sequence_len)}")
    print()


print("Part 4 comparison")

# Function to calculate perplexity
def perplexity(y_true, y_pred):
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.exp(tf.reduce_mean(cross_entropy))

# Evaluate models
rnn_loss, rnn_accuracy = rnn_model.evaluate(X, y, verbose=0)
lstm_loss, lstm_accuracy = lstm_model.evaluate(X, y, verbose=0)

rnn_perplexity = perplexity(y, rnn_model.predict(X)).numpy()
lstm_perplexity = perplexity(y, lstm_model.predict(X)).numpy()

print("RNN Results:")
print(f"Loss: {rnn_loss:.4f}")
print(f"Accuracy: {rnn_accuracy:.4f}")
print(f"Perplexity: {rnn_perplexity:.4f}")

print("\nLSTM Results:")
print(f"Loss: {lstm_loss:.4f}")
print(f"Accuracy: {lstm_accuracy:.4f}")
print(f"Perplexity: {lstm_perplexity:.4f}")

# Compare final validation accuracy
rnn_final_val_acc = rnn_history.history['val_accuracy'][-1]
lstm_final_val_acc = lstm_history.history['val_accuracy'][-1]

print(f"\nFinal Validation Accuracy:")
print(f"RNN: {rnn_final_val_acc:.4f}")
print(f"LSTM: {lstm_final_val_acc:.4f}")

print("Part 6 GPT-2")

import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the pad token to the eos token
tokenizer.pad_token = tokenizer.eos_token

# Assuming we have a WhatsApp corpus file

# Select 5 random partial sentences
partial_sentences = random.sample(whatsApp_corpus, 5)

# Function to generate completion
def generate_completion(partial_sentence):
    inputs = tokenizer(partial_sentence, return_tensors='pt', padding=True, truncation=True)
    output = model.generate(inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_length=50,
                            num_return_sequences=1,
                            no_repeat_ngram_size=2,
                            top_k=50,
                            top_p=0.95,
                            pad_token_id=tokenizer.eos_token_id)
    completed_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
    return completed_sentence

# Generate and print completions
for i, sentence in enumerate(partial_sentences, 1):
    partial = sentence.strip()[:30]  # Take first 30 characters as partial sentence
    completion = generate_completion(partial)
    print(f"Partial sentence {i}: {partial}...")
    print(f"Completion: {completion}\n")

print("Part 7 sentiment analysis")

from transformers import pipeline
from collections import Counter
import matplotlib.pyplot as plt

# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Function to analyze sentiment of a text
def analyze_sentiment(text):
    result = sentiment_analyzer(text[:512])[0]  # Limit to 512 tokens
    return result['label']


# Analyze sentiment for each line in the corpus
sentiments = [analyze_sentiment(line.strip()) for line in whatsApp_corpus if line.strip()]

# Count the occurrences of each sentiment
sentiment_counts = Counter(sentiments)

# Calculate percentages
total = sum(sentiment_counts.values())
sentiment_percentages = {k: (v / total) * 100 for k, v in sentiment_counts.items()}

# Print statistics
print("Sentiment Distribution:")
for sentiment, percentage in sentiment_percentages.items():
    print(f"{sentiment}: {percentage:.2f}%")

# Visualize the distribution
plt.figure(figsize=(10, 6))
plt.bar(sentiment_percentages.keys(), sentiment_percentages.values())
plt.title('Sentiment Distribution in Corpus')
plt.xlabel('Sentiment')
plt.ylabel('Percentage')
plt.show()