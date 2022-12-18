
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
import hdbscan
from math import sqrt, floor
from sklearn.cluster import KMeans


import umap.umap_ as umap
import warnings
warnings.filterwarnings('ignore', category=Warning)

import argparse
from preprocess import *
    
class Clustering:
    def __init__(self,clusterMethod=None,k=10):

        """
        :param k: number of topics
        :param method: method chosen for the topic model
        """
        
        self.k = k
        self.corpus = None
        self.cluster_model = None
        self.method = clusterMethod
        print("Cluster method is:", clusterMethod)

        
    def cluster_train(self,vec):
        
       
        
        if self.method == 'spectral':
            number_clusters = floor(sqrt(len(vec)))
            self.cluster_model = SpectralClustering(n_clusters=number_clusters, 
                                assign_labels="discretize",
                                n_neighbors=number_clusters)
        
        elif self.method == 'KMeans':
            
            self.cluster_model = KMeans(self.k)
        
        
        elif self.method == 'agglomerative':
            # dendrogram = sch.dendrogram(sch.linkage(self.vec[method], method='ward'))

            self.cluster_model = AgglomerativeClustering(n_clusters=self.k, affinity='euclidean', linkage='ward')
        
        elif self.method == 'hdbscan':
            self.cluster_model = hdbscan.HDBSCAN(min_samples=10,min_cluster_size=500)
            
        print("vec is:",len(vec))
        print("vec is:",vec.shape)
        self.cluster_model.fit(vec)
        return self.cluster_model
        print("Cluster model is:", self.cluster_model.labels_)
