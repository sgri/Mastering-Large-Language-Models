#!/usr/bin/env python
# Generated from 'c5-Neural Network Architectures for Language Modeling.ipynb' with convert-jupyter-to-plain-python.sh.
# coding: utf-8

# # Neural network architectures for language modeling

# In[2]:


# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb


# In[ ]:


# Define hyperparameters
max_features = 20000 # Number of words to consider as features
maxlen = 200 # Cut texts after this number of words
batch_size = 32 # Number of samples per batch
embedding_dim = 100 # Dimension of word embeddings
hidden_dim = 128 # Dimension of hidden state
dropout = 0.2 # Dropout probability
learning_rate = 0.001 # Learning rate for optimizer
num_epochs = 10 # Number of training epochs


# In[ ]:


# Load and preprocess data

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences with zeros
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

x_train, y_train


# ## Example of simple LSTM neural network

# In[ ]:


# Define LSTM model using the Sequential API

# Create a sequential model
model = keras.Sequential()
# Add an embedding layer
model.add(layers.Embedding(max_features, embedding_dim))
# Add an LSTM layer with dropout
model.add(layers.LSTM(hidden_dim, dropout=dropout))
# Add a dense layer with sigmoid activation for binary classification
model.add(layers.Dense(1, activation='sigmoid'))


# In[ ]:


# Compile and train the model
# Compile the model with binary cross entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])

# Train the model on training data with validation split
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.2) 


# In[ ]:


# Print model summary
print(model.summary())


# In[ ]:


# Evaluate the model on test data
model.evaluate(x_test, y_test)


# ## Example of simple GRU neural network

# In[ ]:


# Define GRU model using the Sequential API
# Create a sequential model
model = keras.Sequential()
model.add(layers.Embedding(max_features, embedding_dim))

# Add a GRU layer with dropout
model.add(layers.GRU(hidden_dim, dropout=dropout))
# Add a dense layer with sigmoid activation for binary classification
model.add(layers.Dense(1, activation='sigmoid'))

# Compile and train the model
# Compile the model with binary cross entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])

# Train the model on training data with validation split
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.2)

# Print model summary
print(model.summary())

# Evaluate the model on test data
model.evaluate(x_test, y_test) # Evaluate the model on test data


# ## Example of Bidirectional RNN neural network

# In[ ]:


# Define Bidirectional RNN model using the Sequential API
# Create a sequential model
model = keras.Sequential()
model.add(layers.Embedding(max_features, embedding_dim))

# Add a Bidirectional RNN layer with dropout
model.add(layers.Bidirectional(layers.SimpleRNN(hidden_dim, dropout=dropout)))
# Add a dense layer with sigmoid activation for binary classification
model.add(layers.Dense(1, activation='sigmoid'))

# Compile and train the model
# Compile the model with binary cross entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])

# Train the model on training data with validation split
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.2)

# Print model summary
print(model.summary())

# Evaluate the model on test data
model.evaluate(x_test, y_test) # Evaluate the model on test data


# ## Example of CNN network

# In[4]:


from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# Example usage
vocab_size = 10000
embedding_dim = 100
num_filters = 128
filter_size = 3
hidden_dim = 256
output_dim = 10
dropout = 0.5
max_sequence_length = 100

# Define the model architecture
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
model.add(Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dim, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(output_dim, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()


# In[ ]:




