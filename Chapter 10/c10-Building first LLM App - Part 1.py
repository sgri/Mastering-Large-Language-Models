#!/usr/bin/env python
# Generated from 'c10-Building first LLM App - Part 1.ipynb' with convert-jupyter-to-plain-python.sh.
# coding: utf-8

# In[1]:


import openai
from langchain.llms import OpenAI
from langchain.agents import create_csv_agent


# In[2]:


MODEL_NAME = "text-davinci-003"
OPENAI_API_KEY = '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'
openai.api_key = OPENAI_API_KEY


# In[3]:


agent = create_csv_agent(OpenAI(openai_api_key=OPENAI_API_KEY, 
                                     model_name=MODEL_NAME),
                         ['AAPL.csv'], verbose=True)


# In[4]:


agent.agent.llm_chain.prompt.template


# In[5]:


agent.run("How many rows of data do you have?")


# In[6]:


agent.run("What is average high for Apple in year 2022?")


# In[7]:


agent.run("What is the highest high?")


# In[8]:


agent.run("What is the lowest low?")


# In[9]:


agent.run("What is the difference between highest high & lowest low?")


# In[10]:


agent.run("Show me line chart for High, Low over dates")


# In[ ]:




