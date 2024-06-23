#!/usr/bin/env python
# Generated from 'c3-Data Collection and Pre-Processing for Language Modeling.ipynb' with convert-jupyter-to-plain-python.sh.
# coding: utf-8

# # Data Cleaning Techniques - Basic

# **Example 1: Basic Data Cleaning**

# In[1]:


import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import Spellchecker as SpellChecker

# Lowercasing
def lowercase(text):
    text = text.lower()
    return text

# Punctuation removal
def remove_punctuation(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# Removing special characters
def remove_special_chars(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Stop words removal
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return " ".join(filtered_tokens)

# Text standardization
def standardize_text(text):
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    standardized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(standardized_tokens)

# Spelling correction
def correct_spelling(text):
    corrected_text = ""
    from spellchecker import Spellchecker as SpellChecker
    spell = SpellChecker()
    corrected_text = ' '.join([spell.correction(word) for word in text.split()])
    return corrected_text


# In[2]:


# Apply data cleaning techniques
def clean_text(text):
    cleaned_text = lowercase(text)
    cleaned_text = remove_punctuation(cleaned_text)
    cleaned_text = remove_special_chars(cleaned_text)
    cleaned_text = remove_stopwords(cleaned_text)
    cleaned_text = standardize_text(cleaned_text)
    cleaned_text = correct_spelling(cleaned_text)
    return cleaned_text


# In[3]:


# Try on some of the samples
text = "Hello, World! This is an examplle of text cleaning using Python."
cleaned_text = clean_text(text)
print(cleaned_text)

text = "Hi! I'm Sanket and I'm a Data Scientist. I love working with #data and #NLPs. I a have large experience in this fielld."
cleaned_text = clean_text(text)
print(cleaned_text)


# # Data Cleaning Techniques - Advanced

# **Example 2: Entity detection using spaCy library in Python**

# In[4]:


# Import libraries
import spacy
import pandas as pd

# Load a pre-trained spaCy model for English
nlp = spacy.load("en_core_web_sm")

# Define a sample sentence
sentence = "Michael Jackson came to India in 1996 for a concert in Mumbai."

# Apply the spaCy model to the sentence
doc = nlp(sentence)

# Print the entities and their labels
for ent in doc.ents:
    print(ent.text, ent.label_)


# **Example 3: Anonymization using the spaCy library in Python**

# In[5]:


import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Define the text to be anonymized
text = "Michael Jackson came to India in 1996 for a concert in Mumbai."

# Process the text with spaCy
doc = nlp(text)

# Iterate over the named entities and replace them with placeholders
anonymized_tokens = []
for token in doc:
    if token.ent_type_ in ['PERSON', 'GPE', 'DATE']:
        anonymized_tokens.append(token.ent_type_)
    else:
        anonymized_tokens.append(token.text)

# Join the anonymized tokens back into a single string
anonymized_text = ' '.join(anonymized_tokens)

# Print the anonymized text
print(anonymized_text)


# # Text Pre-processing: Preparing Text for Analysis

# **Example 4: Text pre-processing example**

# In[6]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Sample text
text = "The quick brown foxes jumped over the lazy dogs."

# Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in tokens]
print("Stemmed Tokens:", stemmed_tokens)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
print("Lemmatized Tokens:", lemmatized_tokens)


# # Data Annotation

# **Example 5: Part-of-Speech (POS) Tags**

# In[7]:


import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("John wants to buy $1 million house")
for token in doc:
    print(token.text, token.pos_, token.tag_)


# In[8]:


import spacy

print(spacy.explain("NNP"))
print(spacy.explain("VBZ"))


# **Example 6: Dependency parsing**

# In[ ]:


import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("John saw a flashy blue hat at the store")
displacy.serve(doc, style="dep", port=5051)


# In[ ]:




