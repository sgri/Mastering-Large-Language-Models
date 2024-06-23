#!/usr/bin/env python
# Generated from 'Chapter 7.ipynb' with convert-jupyter-to-plain-python.sh.
# coding: utf-8

# In[2]:


get_ipython().system('pip install -q transformers')


# # Training Large Language Models

# ## Building a Tiny Language Model

# In[3]:


from collections import defaultdict

# Function to compute bigram probabilities from a given corpus
def compute_bigram_probabilities(corpus):
    # Create a dictionary to store the bigram counts
    bigram_counts = defaultdict(lambda: defaultdict(int))
    # defaultdict creates missing items instead of throwing KeyError like dict

    # Iterate over each sentence in the corpus
    for sentence in corpus:
        words = sentence.split()  # Tokenize the sentence into words

        # Iterate over each word pair and update the bigram counts
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            bigram_counts[current_word][next_word] += 1

    # Create a dictionary to store the bigram probabilities
    bigram_probabilities = defaultdict(lambda: defaultdict(float))

    # Iterate over each word and its following word in the bigram counts
    for current_word, next_words in bigram_counts.items():
        total_count = sum(next_words.values())
        for next_word, count in next_words.items():
            bigram_probabilities[current_word][next_word] = count / total_count

    return bigram_probabilities

# Input corpus
corpus = ["Peter is happy",
          "Anna is happy",
          "Anna is sad",
          "Anna is good"]

# Compute bigram probabilities
bigram_probabilities = compute_bigram_probabilities(corpus)

# Create dictionaries to store the highest and lowest probabilities for each word
highest_probabilities = {}
lowest_probabilities = {}

# Iterate over each word and its following word in the bigram probabilities
for current_word, next_words in bigram_probabilities.items():
    # Find the word with the highest probability for the current word
    highest_probability = max(next_words, key=next_words.get)
    highest_probabilities[current_word] = highest_probability

    # Find the word with the lowest probability for the current word
    lowest_probability = min(next_words, key=next_words.get)
    lowest_probabilities[current_word] = lowest_probability

# Generate a 2-word sentence beginning with the prompt
prompt = "Peter"
hword = prompt  # for high-probability sentence
lword = prompt  # for low-probability sentence
hsentence = prompt + " "
lsentence = prompt + " "

# Generate the highest-probability and lowest-probability sentences
for _ in range(2):
    hword = highest_probabilities[hword]
    hsentence += hword + " "
    lword = lowest_probabilities[lword]
    lsentence += lword + " "

# Print the generated sentences
print("Highest-probability sentence:", hsentence)
print("Lowest-probability sentence:", lsentence)


# ## Building a Character-level Text Generation Model

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFAutoModelForCausalLM, AutoTokenizer


# Set a seed for reproducibility
tf.random.set_seed(42)

# Sample input text to train the model (you can use your own text)
input_text = """
Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data.
"""


# In[ ]:


# Preprocess the text data
chars = sorted(set(input_text))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
num_chars = len(chars)
input_seq_length = 100
step = 1

input_sequences = []
output_chars = []

for i in range(0, len(input_text) - input_seq_length, step):
    input_seq = input_text[i: i + input_seq_length]
    output_seq = input_text[i + input_seq_length]
    input_sequences.append([char_to_idx[char] for char in input_seq])
    output_chars.append(char_to_idx[output_seq])

X = np.array(input_sequences)
y = tf.keras.utils.to_categorical(output_chars, num_classes=num_chars)

# One-hot encode the input data
X_one_hot = np.zeros((X.shape[0], input_seq_length, num_chars))
for i, seq in enumerate(X):
    for j, char_idx in enumerate(seq):
        X_one_hot[i, j, char_idx] = 1

# Build the LSTM text generator model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(input_seq_length, num_chars)),
    tf.keras.layers.Dense(num_chars, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Training the model
epochs = 100
batch_size = 64

model.fit(X_one_hot, y, epochs=epochs, batch_size=batch_size)


# In[ ]:


# Generate text using the trained model
def generate_text(model, seed_text, num_chars_to_generate=100):
    generated_text = seed_text
    for _ in range(num_chars_to_generate):
        x_pred = np.zeros((1, input_seq_length, num_chars))
        for t, char in enumerate(generated_text[-input_seq_length:]):
            x_pred[0, t, char_to_idx[char]] = 1.0

        preds = model.predict(x_pred, verbose=0)[0]
        next_char_idx = np.argmax(preds)
        next_char = idx_to_char[next_char_idx]
        generated_text += next_char
    return generated_text


# In[ ]:


# Test the text generation
seed_text = "Data science is"
generated_text = generate_text(model, seed_text, num_chars_to_generate=200)
print(generated_text)


# ## Improving Model with Word Tokenization

# In[ ]:


# Preprocess the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([input_text])

# Vocabulary size (number of unique words in the input text)
num_words = len(tokenizer.word_index) + 1

# Convert text to sequences of word indices
input_sequences = tokenizer.texts_to_sequences([input_text])[0]
input_seq_length = 10  # The number of words in each input sequence
step = 1

# Prepare input sequences and output words
X = []
y = []
for i in range(0, len(input_sequences) - input_seq_length, step):
    input_seq = input_sequences[i : i + input_seq_length]
    output_seq = input_sequences[i + input_seq_length]
    X.append(input_seq)
    y.append(output_seq)

X = np.array(X)
y = np.array(y)

# Build the LSTM text generator model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=num_words, output_dim=100, input_length=input_seq_length),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(num_words, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Training the model
epochs = 100
batch_size = 64

model.fit(X, y, epochs=epochs, batch_size=batch_size)



# In[ ]:


# Generate text using the trained model
def generate_text(model, seed_text, num_words_to_generate=50):
    generated_text = seed_text
    for _ in range(num_words_to_generate):
        input_seq = tokenizer.texts_to_sequences([generated_text])[0]
        input_seq = pad_sequences([input_seq], maxlen=input_seq_length, padding='pre')

        preds = model.predict(input_seq)[0]
        next_word_idx = np.argmax(preds)
        next_word = tokenizer.index_word[next_word_idx]
        generated_text += " " + next_word
    return generated_text

# Test the text generation
seed_text = "Data science is"
generated_text = generate_text(model, seed_text, num_words_to_generate=50)
print(generated_text)


# Take 3: Try on larger dataset

# In[ ]:


import requests

response = requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
shakespeare_data = response.text


# In[ ]:


# Sample input text to train the model (you can use your own text)
input_text = shakespeare_data

# Preprocess the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([input_text])

# Vocabulary size (number of unique words in the input text)
num_words = len(tokenizer.word_index) + 1

# Convert text to sequences of word indices
input_sequences = tokenizer.texts_to_sequences([input_text])[0]
input_seq_length = 10  # The number of words in each input sequence
step = 1

# Prepare input sequences and output words
X = []
y = []
for i in range(0, len(input_sequences) - input_seq_length, step):
    input_seq = input_sequences[i : i + input_seq_length]
    output_seq = input_sequences[i + input_seq_length]
    X.append(input_seq)
    y.append(output_seq)

X = np.array(X)
y = np.array(y)

# Build the LSTM text generator model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=num_words, output_dim=100, input_length=input_seq_length),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(num_words, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Training the model
epochs = 10
batch_size = 4096

model.fit(X, y, epochs=epochs, batch_size=batch_size)



# In[ ]:


# Generate text using the trained model
def generate_text(model, seed_text, num_words_to_generate=50):
    generated_text = seed_text
    for _ in range(num_words_to_generate):
        input_seq = tokenizer.texts_to_sequences([generated_text])[0]
        input_seq = pad_sequences([input_seq], maxlen=input_seq_length, padding='pre')

        preds = model.predict(input_seq)[0]
        next_word_idx = np.argmax(preds)
        next_word = tokenizer.index_word[next_word_idx]
        generated_text += " " + next_word
    return generated_text

# Test the text generation
seed_text = "He had rather see the swords"
generated_text = generate_text(model, seed_text, num_words_to_generate=20)
print(generated_text)


# In[ ]:




