import re
import pickle
import requests
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import CountVectorizer

"""
# Fake news detector 
A web app for detecting whether a news article is fake or not
"""
st.markdown("#")
st.write("Enter the link of the article you wish to inspect")

with open("selected_model","rb") as f:
    model=pickle.load(f)
    
with open("vectorizer","rb") as f:
    cv=pickle.load(f)
    
article_link=st.text_input("")

if st.button("Check"):
    if article_link=="":
        st.write("Please input a link.")
    else:
        article=requests.get(article_link)
        soup = BeautifulSoup(article.text, 'html.parser')
        article_title=soup.find("title").text
        article_title=article_title.split('|')[0]
        article_title=article_title.split('-')[0]
        ps=PorterStemmer()
        corpus=[]
        review=re.sub('[^a-zA-Z]',' ',article_title)
        review=review.lower()
        review=review.split()
        review=[ps.stem(word) for word in review if word not in stopwords.words('english')]
        review=' '.join(review)
        corpus.append(review)
        x=cv.fit_transform(corpus).toarray()
        i,j=x.shape
        X=np.zeros((i,5000))
        X[:i,:j]=x
        y=model.predict(X)
        if y[0]==0:
            st.write("This article is a reliable source of information")
        else:
            st.write("This article contains fake news")
