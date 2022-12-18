from main import *


# This file contains the code to visualize the results of the clustering methods on the dataset.
# It basically plots the UMAP projection of the data points and colors them according to the cluster labels.


# Topic Modeling Visualization such as Topic Coherence, Topic Diversity
# Topic Similarity Matrix and Topic Word Cloud Visualization using pyLDAvis and pyLDAvis.sklearn packages 

    
#Importing the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models import LdaModel
from gensim.models import LdaMulticore
from gensim.models import LsiModel
from gensim.models import HdpModel
from gensim.models import TfidfModel

    
#Importing the required packages
import pyLDAvis
import pyLDAvis.gensim
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan
import gensim
import gensim.corpora as corpora

    
def plot_umap(data, labels, title):
    """
    Plot the UMAP projection of the data points and colors them according to the cluster labels.
    :param data: data points
    :param labels: cluster labels
    :param title: title of the plot
    :return: None
    """
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)
    plt.figure(figsize=(7, 5))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=0.1, cmap='Spectral')
    plt.setp(plt.gca(), xticks=[], yticks=[])
    plt.title(title, fontsize=18)
    plt.show()
    
def plot_coherence(model,cv_bow,cv_tfidf,npmi_bow,npmi_tfidf):
    
    model_name = str(model.emebeddingmethod)
    
    pass
    
    
