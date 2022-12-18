
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

import umap.umap_ as umap
import warnings
warnings.filterwarnings('ignore', category=Warning)

import argparse
from preprocess import *

  
# define model object
class Embedding:
    def __init__(self,embeddingmethod=None,embeddingmodel=None,k=10):

        """
        :param k: number of topics
        :param method: method chosen for the topic model
        """
        
        
        self.k = k
        self.dictionary = None
        self.corpus = None
        self.cluster_model = None
        self.ldamodel = None
        self.vec = {}
        self.gamma = 15  
        self.embedding_model = embeddingmodel
        self.emebeddingmethod = embeddingmethod
        
        if embeddingmethod not in {'LDA_S_TRANS', 'LDA_DIFFCSE'}:
            raise Exception('Invalid embeddingmethod!')

    def vectorize(self, sentences, token_lists, method=None):
        """
        Get vecotr representations from selected methods
        """
        # Default method
        if method is None:
            method = self.emebeddingmethod

        # turn tokenized documents into a id <-> term dictionary
        self.dictionary = corpora.Dictionary(token_lists)
        # convert tokenized documents into a document-term matrix
        self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]


        if method == 'LDA':
            print('Getting vector representations for LDA ...')
            if not self.ldamodel:
                self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                                passes=20)

            def get_vec_lda(model, corpus, k):
                """
                Get the LDA vector representation (probabilistic topic assignments for all documents)
                :return: vec_lda with dimension: (n_doc * n_topic)
                """
                n_doc = len(corpus)
                vec_lda = np.zeros((n_doc, k))
                for i in range(n_doc):
                    for topic, prob in model.get_document_topics(corpus[i]):
                        vec_lda[i, topic] = prob

                return vec_lda

            vec = get_vec_lda(self.ldamodel, self.corpus, self.k)
            print('Getting vector representations for LDA. Done!')
            
            return vec

        elif method == 'S_TRANS':

            print('Getting vector representations for Sentence Transformer ...')
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer(self.embedding_model) # Change the embedding model(Pre-trained model)
            vec = np.array(model.encode(sentences, show_progress_bar=True))
            print('Getting vector representations for Sentence Transformer. Done!')
            
            return vec
             
        elif method == 'DIFFCSE':
            print("Generating DiffCSE Sentence Embedding....")
            from DiffCSE.diffcse import DiffCSE
            model = DiffCSE(self.embedding_model) # Change the embedding model(Pre-trained model
            vec_diffcse = model.encode(sentences)
            print("Getting Vector representations for DiffCSE embedding")
            
            return vec_diffcse
        
        elif method == 'LDA_S_TRANS':
 
            vec_lda = self.vectorize(sentences, token_lists, method='LDA')
            vec_bert = self.vectorize(sentences, token_lists, method='S_TRANS')
            vec_ldabert = np.c_[vec_lda * self.gamma, vec_bert]
            self.vec['LDA_BERT_FULL'] = vec_ldabert
        
            return vec_ldabert

        elif method == 'LDA_DIFFCSE':
  
            vec_lda = self.vectorize(sentences, token_lists, method='LDA')
            from DiffCSE.diffcse import DiffCSE
            model = DiffCSE("voidism/diffcse-bert-base-uncased-sts")
            vec_diffcse = model.encode(sentences)
            vec_ldadiffcse = np.c_[vec_lda * self.gamma, vec_diffcse]
            print("LDA+DIFFCSE Shape:", vec_ldadiffcse.shape)
            self.vec['LDA_BERT_FULL'] = vec_ldadiffcse
           
            return vec_ldadiffcse
        
    

    
    
