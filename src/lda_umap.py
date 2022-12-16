
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
import hdbscan
import os
from wordcloud import WordCloud

import umap.umap_ as umap
import warnings
warnings.filterwarnings('ignore', category=Warning)

import argparse
from preprocess import *

# define model object
class Topic_Model:
    def __init__(self, k=10, method='TFIDF'):
        """
        :param k: number of topics
        :param method: method chosen for the topic model
        """
        if method not in {'TFIDF', 'LDA', 'BERT', 'LDA_BERT','SROBERTA'}:
            raise Exception('Invalid method!')
        self.k = k
        self.dictionary = None
        self.corpus = None
        #         self.stopwords = None
        self.cluster_model = None
        self.ldamodel = None
        self.vec = {}
        self.gamma = 15  # parameter for reletive importance of lda
        self.method = method
        self.AE = None
        self.id = method + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def vectorize(self, sentences, token_lists, method=None):
        """
        Get vecotr representations from selected methods
        """
        # Default method
        if method is None:
            method = self.method
        print("HERE HERE")
        # turn tokenized documents into a id <-> term dictionary
        self.dictionary = corpora.Dictionary(token_lists)
        # convert tokenized documents into a document-term matrix
        self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]

        if method == 'TFIDF':
            print('Getting vector representations for TF-IDF ...')
            tfidf = TfidfVectorizer()
            vec = tfidf.fit_transform(sentences)
            print('Getting vector representations for TF-IDF. Done!')
            return vec

        elif method == 'LDA':
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
                # print("n_doc:)", n_doc)
                # print("n_doc",n_doc)
                vec_lda = np.zeros((n_doc, k))
                # print("vec_lda shape", vec_lda.shape)
                for i in range(n_doc):
                    # get the distribution for the i-th document in corpus
                    for topic, prob in model.get_document_topics(corpus[i]):
                        vec_lda[i, topic] = prob
                # print("corpus:", len(corpus))

                #print("vec_lda shape:",vec_lda)
                return vec_lda

            vec = get_vec_lda(self.ldamodel, self.corpus, self.k)
            print('Getting vector representations for LDA. Done!')
            return vec

        elif method == 'BERT':

            print('Getting vector representations for BERT ...')
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('bert-base-nli-max-tokens')
            vec = np.array(model.encode(sentences, show_progress_bar=True))
            print('Getting vector representations for BERT. Done!')
            print("BERT Vec shapoe:", vec.shape)
            print("BERT sebetnces:", len(sentences))
            return vec
        elif method == 'SROBERTA':
            from transformers import AutoTokenizer, AutoModelForMaskedLM
  
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            #all-distilroberta-v1
            # tokenizer = AutoTokenizer.from_pretrained("Andrija/SRoBERTa")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-distilroberta-v1')
            print('Getting vector representations for SROBERTA ...')
            # model = AutoModelForMaskedLM.from_pretrained("Andrija/SRoBERTa")
            vec = np.array(model.encode(sentences, show_progress_bar=True))
            print('Getting vector representations for S-RoBERTA. Done!')
            
            
            # Kritika - added for roberta + umap
            clusterable_embedding = umap.UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=32,
            random_state=42,
            ).fit_transform(vec)
            
            print('Getting umap represntation for S-RoBERTA. Done!')
            print('Getting vector representations for S-RoBERTA. Done!')
            return clusterable_embedding

             
        elif method == 'LDA_BERT':
        #else:
            # print("Senetnces",len(sentences))
            # print("token_lists",len(token_lists))
            vec_lda = self.vectorize(sentences, token_lists, method='LDA')
            vec_bert = self.vectorize(sentences, token_lists, method='BERT')
            # print("vec_lda shape:", vec_lda.shape)
            # print("vec_bert shape:", vec_bert.shape)
            vec_ldabert = np.c_[vec_lda * self.gamma, vec_bert]
            print("vec_vec:", vec_ldabert.shape)
            
            
            self.vec['LDA_BERT_FULL'] = vec_ldabert
            
            # Kritika -
            # if not self.AE:
            #     self.AE = Autoencoder()
            #     print('Fitting Autoencoder ...')
            #     self.AE.fit(vec_ldabert)
            #     print('Fitting Autoencoder Done!')
            # vec = self.AE.encoder.predict(vec_ldabert)
            # print("Autoencoder vec_ldabert:", vec.shape)
            
            clusterable_embedding = umap.UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=64,
            random_state=42,
            ).fit_transform(vec_ldabert)
            
            
            # labels = hdbscan.HDBSCAN(
            # min_samples=10,
            # min_cluster_size=500,
            # ).fit_predict(clusterable_embedding)
            print("Executing UMAP code")
            print("Embedding shape:", clusterable_embedding.shape)

            
            return clusterable_embedding

    def fit(self, sentences, token_lists, method=None, m_clustering=None):
        """
        Fit the topic model for selected method given the preprocessed data
        :docs: list of documents, each doc is preprocessed as tokens
        :return:
        """
        # Default method
        if method is None:
            method = self.method
        # Default clustering method
        if m_clustering is None:
            m_clustering = KMeans

        # turn tokenized documents into a id <-> term dictionary
        if not self.dictionary:
            self.dictionary = corpora.Dictionary(token_lists)
            # convert tokenized documents into a document-term matrix
            self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]
            # print("corpus:",self.corpus)
            print("corpus size:",len(self.corpus))

        ####################################################
        #### Getting ldamodel or vector representations ####
        ####################################################

        if method == 'LDA':
            if not self.ldamodel:
                print('Fitting LDA ...')
                self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                                passes=20)
                print('Fitting LDA Done!')
        else:
            print('Clustering embeddings ...')
            
            
            
            self.vec[method] = self.vectorize(sentences, token_lists, method)
            
            # UnComment for kmeans
            self.hdb_model = hdbscan.HDBSCAN(
            min_samples=10,
            min_cluster_size=500,
            )
            self.hdb_model.fit(self.vec[method])
            
            # Comment for hdb scan
            # self.cluster_model = m_clustering(self.k)
            # self.cluster_model.fit(self.vec[method])
            
            print("self.vec[method] shape:", self.vec[method].shape)
            # self.c_model = m_clustering(self.k)
            
            print('Clustering embeddings. Done!')
            # print("Labels are:", labels.labels_)
    def predict(self, sentences, token_lists, out_of_sample=None):
        """
        Predict topics for new_documents
        """
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
    
    