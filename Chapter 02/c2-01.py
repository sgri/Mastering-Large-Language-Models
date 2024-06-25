#!/usr/bin/env python
# Generated from 'c2-Introduction to Language Models.ipynb' with convert-jupyter-to-plain-python.sh.
# coding: utf-8

# # Training 'Rule-Based Models'

# **Example 1: Using regular expressions**

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
