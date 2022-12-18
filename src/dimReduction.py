
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
from collections import Counter
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import os
from wordcloud import WordCloud

import umap.umap_ as umap
import warnings
warnings.filterwarnings('ignore', category=Warning)

import argparse
from preprocess import *

class Autoencoder:
  """
  Autoencoder for learning latent space representation
  architecture simplified for only one hidden layer
  """
  def __init__(self, latent_dim = 32, activation='relu', epochs=200, batch_size=128):
    self.latent_dim = latent_dim
    self.activation = activation
    self.epochs = epochs
    self.batch_size = batch_size
    self.autoencoder = None
    self.encoder = None
    self.decoder = None
    self.his = None

  def _compile(self, input_dim):
    '''
    compile the computational graph
    '''
    input_vec = Input(shape=(input_dim,))
    encoded = Dense(self.latent_dim, activation=self.activation)(input_vec)
    decoded = Dense(input_dim, activation=self.activation)(encoded)
    self.autoencoder = Model(input_vec, decoded)
    self.encoder = Model(input_vec, encoded)
    encoder_input = Input(shape=(self.latent_dim,))
    decoded_layer = self.autoencoder.layers[-1]
    self.decoder = Model(encoder_input, self.autoencoder.layers[-1](encoder_input))
    self.autoencoder.compile(optimizer='adam', loss=keras.losses.mean_absolute_error)

  def fit(self, X):
    if not self.autoencoder:
      self._compile(X.shape[1])
    X_train, X_test = train_test_split(X)
    self.his = self.autoencoder.fit(X_train, X_train, epochs=200, batch_size=128, shuffle=True, validation_data=(X_test, X_test), verbose=0)
    
class Reduction():

    def __init__(self, dimReductionMethod=None):
        self.AE = None
        self.method = dimReductionMethod
    
    def fit(self,vec):
        print("Dimensionality Reduction Method is :", self.method)
        print("Clustering Embedding Shape: ", vec.shape)
        if self.method == 'UMAP':
            clusterable_embedding = umap.UMAP(
                n_neighbors=30,
                min_dist=0.0,
                n_components=32,
                random_state=42,
                ).fit_transform(vec)
            print("Clustering Embedding Shape: ", clusterable_embedding.shape)
            
            return clusterable_embedding
            
        elif self.method == 'AE':
            self.AE = Autoencoder()
            print('Fitting Autoencoder ...')
            self.AE.fit(vec)
            print('Fitting Autoencoder Done!')
            vec_ = self.AE.encoder.predict(vec)

            return vec_