#!/usr/bin/env python
# Generated from 'c2-Introduction to Language Models.ipynb' with convert-jupyter-to-plain-python.sh.
# coding: utf-8

# # Training 'Rule-Based Models'

# **Example 1: Using regular expressions**

# In[1]:


import spacy
import re

nlp = spacy.load('en_core_web_sm')

# Define a pattern using a regular expression
pattern = r"\d{3}-\d{3}-\d{4}"  # Matches phone numbers in the format XXX-XXX-XXXX

# Text to be processed
text = "Please call me at 123-456-7890."

# Process the text
doc = nlp(text)

# Iterate over the matches
for match in re.finditer(pattern, doc.text):
    start, end = match.span()
    span = doc.char_span(start, end)
    # This is a Span object or None if match doesn't map to valid token sequence
    if span is not None:
        print("Found match:", span.text)


# **Example 2: Matching patterns using part-of-speech tags**

# In[2]:


import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

# Define the pattern using part-of-speech tags
pattern = [{'POS': 'NOUN'}, {'POS': 'VERB'}]

# Add the pattern to the matcher
matcher.add('NOUN_VERB_NOUN_PATTERN', [pattern])

# Text to be processed
text = "I saw a boy playing in the gardan."

# Process the text
doc = nlp(text)

# Apply the matcher to the doc
matches = matcher(doc)

# Iterate over the matches
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)


# **Example 3: Matching specific tokens with specific attributes**

# In[3]:


import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

# Define the pattern
pattern = [{'LOWER': 'machine'}, {'LOWER': 'learning'}]

# Add the pattern to the matcher
matcher.add('ML_PATTERN', [pattern])

# Text to be processed
text = "I am interested in machine learning and deep learning."

# Process the text
doc = nlp(text)

# Apply the matcher to the doc
matches = matcher(doc)

# Iterate over the matches
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)


# # Training 'Statistical Models'

# **Example 4: simple implementation of a bigram language model.**

# In[4]:


# Define a function to read the data from a list of sentences
def readData():
    # Initialize an empty list to store the words
    data = ['This is a house','This is a home','I love my house','This is my home', 'Is this your house?']
    dat=[]
    # Loop through each sentence in the data
    for i in range(len(data)):
        # Split the sentence into words and append them to the list
        for word in data[i].split():
            dat.append(word)
    # Print the list of words
    print(dat)
    # Return the list of words
    return dat

# Define a function to create bigrams from the list of words
def createBigram(data):
   # Initialize an empty list to store the bigrams
   listOfBigrams = []
   # Initialize an empty dictionary to store the bigram counts
   bigramCounts = {}
   # Initialize an empty dictionary to store the unigram counts
   unigramCounts = {}
   # Loop through each word in the data except the last one
   for i in range(len(data)-1):
      # Check if the next word is lowercase (to avoid punctuation marks)
      if i < len(data) - 1 and data[i+1].islower():
         # Create a bigram tuple from the current and next word and append it to the list
         listOfBigrams.append((data[i], data[i + 1]))
         # Increment the count of the bigram in the dictionary or set it to 1 if not present
         if (data[i], data[i+1]) in bigramCounts:
            bigramCounts[(data[i], data[i + 1])] += 1
         else:
            bigramCounts[(data[i], data[i + 1])] = 1
      # Increment the count of the current word in the dictionary or set it to 1 if not present
      if data[i] in unigramCounts:
         unigramCounts[data[i]] += 1
      else:
         unigramCounts[data[i]] = 1
   # Return the list of bigrams, the unigram counts and the bigram counts
   return listOfBigrams, unigramCounts, bigramCounts


# Define a function to calculate the bigram probabilities from the counts
def calcBigramProb(listOfBigrams, unigramCounts, bigramCounts):
    # Initialize an empty dictionary to store the bigram probabilities
    listOfProb = {}
    # Loop through each bigram in the list
    for bigram in listOfBigrams:
        # Get the first and second word of the bigram
        word1 = bigram[0]
        word2 = bigram[1]
        # Calculate the probability of the bigram as the ratio of its count and the count of the first word
        listOfProb[bigram] = (bigramCounts.get(bigram))/(unigramCounts.get(word1))
    # Return the dictionary of bigram probabilities
    return listOfProb


# Call the readData function and store the result in data variable
data = readData()
# Call the createBigram function with data as argument and store the results in three variables
listOfBigrams, unigramCounts, bigramCounts = createBigram(data)

# Print some messages and results for debugging purposes
print("\n All the possible Bigrams are ")
print(listOfBigrams)

print("\n Bigrams along with their frequency ")
print(bigramCounts)

print("\n Unigrams along with their frequency ")
print(unigramCounts)

# Call the calcBigramProb function with the counts as arguments and store the result in bigramProb variable
bigramProb = calcBigramProb(listOfBigrams, unigramCounts, bigramCounts)

print("\n Bigrams along with their probability ")
print(bigramProb)


# **Example 5: Example of find ngrams of the sentence**

# In[5]:


import nltk
from nltk.util import ngrams

# Function to generate n-grams from sentences.
def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [ ' '.join(grams) for grams in n_grams]

My_text = 'I am interested in machine learning and deep learning.'

print("1-gram of the sample text: ", extract_ngrams(My_text, 1), '\n')
print("2-gram of the sample text: ", extract_ngrams(My_text, 2), '\n')
print("3-gram of the sample text: ", extract_ngrams(My_text, 3), '\n')
print("4-gram of the sample text: ", extract_ngrams(My_text, 4), '\n')


# In[ ]:




