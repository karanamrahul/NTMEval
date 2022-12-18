
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim import corpora
import gensim
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import nltk
nltk.download('punkt')
import nltk
nltk.download('averaged_perceptron_tagger')
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import umap.umap_ as umap
from wordcloud import WordCloud
from collections import Counter
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
import wordcloud as WordCloud
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import os
from wordcloud import WordCloud
from embedding import *
import umap.umap_ as umap
import warnings
warnings.filterwarnings('ignore', category=Warning)

import argparse
from preprocess import *

from embedding import *
from dimReduction import *
from cluster import *


# define model object
class Topic_Model:
    def __init__(self, embeddingmethod, dimReduction=None, clusterMethod=None,embeddingmodel=None,k=10):   
        
        if embeddingmethod not in {'LDA_S_TRANS', 'LDA_DIFFCSE'}:
            raise Exception('Invalid embeddingmethod!')
        if dimReduction not in {'AE', 'UMAP', 'VAE'}:
            raise Exception('Invalid dimReduction!')
        if clusterMethod not in {'KMeans', 'spectral', 'agglomerative', 'hdbscan'}:
            raise Exception('Invalid clusterMethod!')
        
        self.k = k
        self.dictionary = None
        self.corpus = None
        self.cluster_model = None
        self.embeddingmodel = embeddingmodel
        self.ldamodel = None
        self.vec = {}
        self.gamma = 15  
        self.emebeddingmethod = embeddingmethod
        self.dimReduction = dimReduction
        self.clusterMethod = clusterMethod
        self.embed = None
        self.cluster = None
        self.dim = None
        self.AE = None
        
    def fit(self, sentences, token_lists, method=None, m_clustering=None):
            
        if not self.dictionary:
            self.dictionary = corpora.Dictionary(token_lists)
            # convert tokenized documents into a document-term matrix
            self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]
            # print("corpus:",self.corpus)
            print("corpus size:",len(self.corpus))
        print("Embedding method is 3 :", self.emebeddingmethod)
        self.embed = Embedding(self.emebeddingmethod, self.embeddingmodel)
        embedding = self.embed.vectorize(sentences, token_lists, method)
        
        self.dim = Reduction(self.dimReduction)
        
        reduced_embedding = self.dim.fit(embedding)
        
        self.cluster = Clustering(self.clusterMethod)
        
        cluster_model=self.cluster.cluster_train(reduced_embedding)
        return cluster_model
        
    def predict(self, sentences, token_lists, out_of_sample=None):
        
        # Default as False
        out_of_sample = out_of_sample is not None

        if out_of_sample:
            corpus = [self.dictionary.doc2bow(text) for text in token_lists]
            if self.method != 'LDA':
                vec = self.vectorize(sentences, token_lists)
                # print(vec)
        else:
            corpus = self.corpus
            vec = self.vec.get(self.method, None)
        
        print("Vec shape in predict:", vec.shape)

        if self.method == "LDA":
            lbs = np.array(list(map(lambda x: sorted(self.ldamodel.get_document_topics(x),
                                                     key=lambda x: x[1], reverse=True)[0][0],
                                    corpus)))
        else:
            lbs = self.cluster_model.predict(vec)
        return lbs
    