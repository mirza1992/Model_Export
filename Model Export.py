from __future__ import print_function

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import random
import re
import time
import string
nltk.download('stopwords')
nltk.download('punkt')
import pickle
import pyLDAvis
import pyLDAvis.sklearn


def content_input(content):
    content = input("Enter Article:")
    

    def pre_processor(col_name):

      col_name = col_name.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ')

      # 1- Converting to Lower & Clear the Oppostrophy
      col_name = col_name.lower().replace(u'\n', u' ').replace(u"’", u'').replace(u'“', u'').replace(u'”', u'').replace(u'   ', u'')

      # 2- Remove punctuations
      col_name = col_name.replace('[{}]'.format(string.punctuation), '')

      # 4- Remove URL's
      col_name = col_name.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ')


      return col_name

    content_preprocess = pre_processor(content)
    pre_proc = [content_preprocess]


    tfidf = TfidfVectorizer(stop_words='english',ngram_range=(2,2) ,strip_accents=None, lowercase=True, max_features=None, vocabulary=None)
    tfidf.fit(pre_proc)
    vect_tfidf = tfidf.fit_transform(pre_proc)


    LDA_tfidf = LatentDirichletAllocation(n_components=10,           # Number of topics
                                          max_iter=5,               # Max learning iterations
                                          learning_method='online'          
                                         )
    LDA_tfidf.fit(vect_tfidf)



    for index, topic in enumerate(LDA_tfidf.components_):
      print('\n')
      print(f"The Top Words For Topic #{index}")
      print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
      print('\n')
    pyLDAvis.enable_notebook()
    pyLDAvis.sklearn.prepare(LDA_tfidf, vect_tfidf, tfidf)


if __name__ == "__main__":
    content_input(content)
