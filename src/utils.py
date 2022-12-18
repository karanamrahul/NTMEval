
from preprocess import *
from train import *
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from bertTopic import *
from collections import Counter
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from octis.evaluation_metrics.diversity_metrics import InvertedRBO  
from sklearn.metrics.cluster import rand_score  


def get_topic_words(token_lists, labels, k=None):
  ''' Get topic within each topic form clustering results '''
  if k is None:
    k = len(np.unique(labels))
  topics = ['' for _ in range(k)]
  # print("Topics:", topics)
  for i, c in enumerate(token_lists):
    
    topics[labels[i]] += (' ' + ' '.join(c))
  # print("Topics:", topics)
  word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
  # print("word_counts:", len(word_counts))
  # print("word_counts:", word_counts[1:10])
  # get sorted word counts
  word_counts = list(map(lambda x: sorted(x, key=lambda x: x[1], reverse=True),word_counts))
  # print("New word_counts:", len(word_counts))
  # print("New word_counts:", word_counts[1:10])
  # get topics
  topics = list(map(lambda x: list(map(lambda x: x[0], x[:10])), word_counts))
  # print("New topics:", len(topics))
  # print("New topics:", topics[1:10])

  return topics

def get_topic_words_ctfidf(token_lists, labels, k=None):
      
  class CTFIDFVectorizer(TfidfTransformer):
        
      def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer, self).__init__(*args, **kwargs)

      def fit(self, X: sp.csr_matrix, n_samples: int):
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        idf = np.log(n_samples / df)
        self._idf_diag = sp.diags(idf, offsets=0,shape=(n_features, n_features),format='csr',dtype=np.float64)
        return self

      def transform(self, X: sp.csr_matrix) -> sp.csr_matrix:
        X = X * self._idf_diag
        X = normalize(X, axis=1, norm='l1', copy=False)
        return X


  return True

# def get_coherence(model, token_list, measure='c_v'):
#   ''' Get model coherence from gensim.models.coherencemodel
#   : param model: Topic_Model object
#   : param token_lists: token list of docs
#   : param topics: topics as top words 
#   : param measure: coherence metrics
#   : return: coherence score '''

#   if model.method == 'LDA':
#     cm = CoherenceModel(model=model.ldamodel, texts=token_list, corpus = model.corpus, dictionary=model.dictionary, coherence = measure)
#   else:
#     # Comment - kritika
#     # topics = get_topic_words(token_list, model.cluster_model.labels_)
    
#     topics = get_topic_words(token_list, model.hdb_model.labels_)
#     cm = CoherenceModel(topics=topics, texts = token_list, corpus=model.corpus, dictionary=model.dictionary, coherence = measure)
#     return cm.get_coherence()

def get_silhoulette(model):
  ''' Get silhoulette score from model
  :param_model: Topic_model score
  :return: silhoulette score '''
  if model.method == 'LDA':
    return
  lbs = model.cluster_model.labels_
  vec = model.vec[model.method]
  return silhouette_score(vec, lbs)

def plot_project(embeddings, lbs):
  '''
  Plot UMAP embeddings
  :param embedding: UMAP (or other) embeddings
  :param lbs: labels
  '''
  n = len(embeddings)
  counter = Counter(lbs)
  for i in range(len(np.unique(lbs))):
    plt.plot(embeddings[:, 0][lbs == i], embeddings[:, 1][lbs == i], '.', alpha=0.5, label='cluster {}: {:.2f}%'.format(i, counter[i] / n*100))
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='-', linewidth=0.25)

def visualize(model):
  '''
  Visualize the result for the topic model by 2D embedding (UMAP)
  :param model: Topic_Model object
  '''
  if model.method == 'LDA':
    return
  reducer = umap.UMAP()
  print('Calculating UMAP projection...')
  vec_umap = reducer.fit_transform(model.vec[model.method])
  print('Calculating the Umap projection. Done!')
  plot_project(vec_umap, model.cluster_model.labels_)

def get_wordcloud(model, token_list, topics):
  """
  Get word cloud of each topic from fitted model
  :param model: Topic_Model object
  :param sentences: preprocessed sentences from docs
  """
  if model.method == 'LDA':
    return
  print('Getting wordcloud for topic {}... '.format(topics))
  lbs = model.cluster_model.labels_
  tokens = ' '.join([' '.join(_) for _ in np.array(token_list)[lbs == topics]])
  wordcloud = WordCloud(width=800, height=560, background_color='white', collocations=False, min_font_size=10).generate(tokens)
  # plot the WordCloud image
  plt.figure(figsize=(8, 5.6), facecolor=None)
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.tight_layout(pad=0)
  print('Getting wordcloud for topics {}. Done!'.format(topics))
  
  
def get_coherence(model,cluster_model, token_list): 
    # topics = get_topic_words(token_list, model.cluster_model.labels_
    print("model.sub_cluster.labels_", len(cluster_model.labels_))
    topics = get_topic_words(token_list, cluster_model.labels_)
    
    # topics = get_topic_words(token_list, model.hdb_model.labels_)
    cm_cv = CoherenceModel(topics=topics, texts = token_list, corpus=model.corpus, dictionary=model.dictionary, coherence = 'c_v')
    cm_npmi = CoherenceModel(topics=topics, texts = token_list, corpus=model.corpus, dictionary=model.dictionary, coherence = 'c_npmi')
    return cm_cv.get_coherence(), cm_npmi.get_coherence()
      
def get_topic_diversity(model, token_list, k):
      
    """_summary_: Get topic diversity from gensim.models.coherencemodel
    : param model: Topic_Model object
    : param token_lists: token list of docs
    : param topics: topics as top words"""

    topic_keywords = get_topic_words(token_list, model.labels_) # get topic words
    div_dict = {'topics' : topic_keywords}
    div = TopicDiversity(k)
    td = div.score(div_dict)

    return td
      
      
def get_irbo(model, token_list):
    """_summary_: Get irbo from gensim.models.coherencemodel
    : param model: Topic_Model object
    : param token_lists: token list of docs
    : param topics: topics as top words"""
    topic_keywords = get_topic_words(token_list, model.labels_) # get topic words
    irbo_Dict = {'topics' : topic_keywords}
    rbo = InvertedRBO()
    irbo = rbo.score(irbo_Dict)

    return irbo

def get_rand_index(model, labels_true, token_list):
    topic_keywords = get_topic_words(token_list, model.labels_) # get topic words
    labels_pred = model.labels_
    labels_pred_len = len(labels_pred)
    labels_true = labels_true[:labels_pred_len]
    labels_true_len = len(labels_true)
    scores = rand_score(labels_true, labels_pred)
    return scores

def print_evaluations(model, cluster_model, token_lists, labels, k, method):
    coherence_cv , coherence_npmi = get_coherence(model, cluster_model, token_lists)
    diversity = get_topic_diversity(cluster_model, token_lists, k)
    irbo = get_irbo(cluster_model, token_lists)
    # silhoulette = get_silhoulette(model)
    rand_index = get_rand_index(cluster_model, labels, token_lists)
    print("-"*50)
    print("Model       Score")
    print("-"*50)
    print("{} Coherence CV: {}".format(method, coherence_cv))
    print("-"*50)
    print("{} Coherence NPMI: {}".format(method, coherence_npmi))
    print("-"*50)
    print("{} Topic Diversity: {}".format(method, diversity))
    print("-"*50)
    print("{} IRBO: {}".format(method, irbo))
    # print("-"*50)
    # print("{} Silhoulette: {}".format(method, silhoulette))
    print("-"*50)
    print("{} Rand Index: {}".format(method, rand_index)) 


