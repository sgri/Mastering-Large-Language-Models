#!/usr/bin/env python
# Generated from 'c3-Data Collection and Pre-Processing for Language Modeling.ipynb' with convert-jupyter-to-plain-python.sh.
# coding: utf-8

# # Data Annotation

# **Example 5: Part-of-Speech (POS) Tags**

import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("John wants to buy $1 million house")
for token in doc:
    print(token.text, token.pos_, token.tag_)




