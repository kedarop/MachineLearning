#!/usr/bin/env python
# coding: utf-8

# BOOK RECOMMENDATION SYSTEM

# In[3]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import plotly.express as px
import plotly.graph_objects as go


# In[4]:


data = pd.read_csv("books_data.csv")
print(data.head())


# In[5]:


data.info()


# In[6]:


fig = px.histogram(data, x='average_rating', nbins=30, title='Distribution of Average Ratings')
fig.update_xaxes(title_text='Average Rating')
fig.update_yaxes(title_text='Frequency')
fig.show()


# In[7]:


top_authors = data['authors'].value_counts().head(10)
fig = px.bar(top_authors, x=top_authors.values, y=top_authors.index, orientation='h',
             labels={'x': 'Number of Books', 'y': 'Author'},
             title='Number of Books per Author')
fig.show()


# In[8]:


# Convert 'average_rating' to a numeric data type
data['average_rating'] = pd.to_numeric(data['average_rating'], errors='coerce')


# In[9]:


# Create a new column 'book_content' by combining 'title' and 'authors'
data['book_content'] = data['title'] + ' ' + data['authors']


# In[10]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['book_content'])


# In[11]:


# Compute the cosine similarity between books
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[12]:


def recommend_books(book_title, cosine_sim=cosine_sim):
    # Get the index of the book that matches the title
    idx = data[data['title'] == book_title].index[0]

    # Get the cosine similarity scores for all books with this book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 most similar books (excluding the input book)
    sim_scores = sim_scores[1:11]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 recommended books
    return data['title'].iloc[book_indices]


# In[13]:


book_title = "Dubliners: Text  Criticism  and Notes"
recommended_books = recommend_books(book_title)
print(recommended_books)


# In[ ]:




