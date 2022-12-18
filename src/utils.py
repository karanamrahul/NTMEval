
from preprocess import *
from train import *
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from BertTopic import *
from collections import Counter
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from octis.evaluation_metrics.diversity_metrics import InvertedRBO  
from sklearn.metrics.cluster import rand_score  
import numpy as np



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

def get_topic_words_ctfidf(sentences,token_lists, labels, k=None):
      

  class CTFIDFVectorizer(TfidfTransformer):
    def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer, self).__init__(*args, **kwargs) # call parent constructor

    def fit(self, X: sp.csr_matrix, n_samples: int): # X is a sparse matrix of counts (n_samples, n_features)
        """Learn the idf vector (global term weights) """
        _, n_features = X.shape # n_features is the number of words in the vocabulary
        df = np.squeeze(np.asarray(X.sum(axis=0))) # df is the number of documents in which each word appears
        idf = np.log(n_samples / df)  # idf is the inverse document frequency
        self._idf_diag = sp.diags(idf, offsets=0, 
                                  shape=(n_features, n_features),
                                  format='csr',
                                  dtype=np.float64) # idf as a diagonal matrix
        return self

    def transform(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """Transform a count-based matrix to c-TF-IDF """
        X = X * self._idf_diag # element-wise multiplication
        X = normalize(X, axis=1, norm='l1', copy=False) # l1-normalize each row
        return X # return c-TF-IDF matrix
      
  # Using the predicted labels and sentences to get the topics based on the c-TF-IDF method 
  
  # Need to merge the token_lists and labels into a dataframe
  print("Length of Clusters", len(labels))
  df = pd.DataFrame({'data': sentences, 'labels': labels})
  df_per_class = df.groupby('labels').agg({'data': ' '.join})
  
  print("df_per_class",len(df_per_class))
  print("df_per_class",len(df))
  count = CountVectorizer().fit_transform(df_per_class.data)
  ctfidf = CTFIDFVectorizer().fit(count, len(df))
  words  = CountVectorizer().fit(df_per_class.data).get_feature_names()
  
  # Get the top 5 words for each topic
  topics = []
  for i in range(len(df_per_class.data)):
    topic = ctfidf.transform(count[i])
    
    print("topic", topic)
    topic = topic.toarray()[0]
    topic = np.argsort(topic)[::-1][:10]
    topics.append(topic)
  
  
  # Convert the topics into words
  topics = [[words[i] for i in topic] for topic in topics]
  
  # # Extract the top 10 words for each class
  # words_per_class = { i: topics[i] for i in range(0, len(topics) ) } 



  
  
  return topics




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
  
  
def get_coherence(model,cluster_model, token_list,sentences): 
    # topics = get_topic_words(token_list, model.cluster_model.labels_
    print("model.sub_cluster.labels_", len(cluster_model.labels_))
    # topics = get_topic_words(token_list, cluster_model.labels_)
    
    topics_bow=get_topic_words(token_list,cluster_model.labels_)
    topics_ctfidf=get_topic_words_ctfidf(sentences,token_list,cluster_model.labels_)
    # print("Topics ",topics)
    # print("Cluster model labels", cluster_model.labels_)
    
    # topics = get_topic_words(token_list, model.hdb_model.labels_)
    cm_cv_bow = CoherenceModel(topics=topics_bow, texts = token_list, corpus=model.corpus, dictionary=model.dictionary, coherence = 'c_v')
    cm_npmi_bow = CoherenceModel(topics=topics_bow, texts = token_list, corpus=model.corpus, dictionary=model.dictionary, coherence = 'c_npmi')
    cm_cv_tfidf = CoherenceModel(topics=topics_ctfidf, texts = token_list, corpus=model.corpus, dictionary=model.dictionary, coherence = 'c_v')
    cm_npmi_tfidf = CoherenceModel(topics=topics_ctfidf, texts = token_list, corpus=model.corpus, dictionary=model.dictionary, coherence = 'c_npmi')
    
    return cm_cv_bow.get_coherence(), cm_npmi_bow.get_coherence(),cm_cv_tfidf.get_coherence(), cm_npmi_tfidf.get_coherence()
      
def get_topic_diversity(model, token_list, k,sentences):
      
    """_summary_: Get topic diversity from gensim.models.coherencemodel
    : param model: Topic_Model object
    : param token_lists: token list of docs
    : param topics: topics as top words"""

    topic_keywords_bow = get_topic_words(token_list, model.labels_) # get topic words
    topic_keywords_tfidf = get_topic_words_ctfidf(sentences,token_list, model.labels_) # get topic words
    
    div = TopicDiversity(k)
    
    div_dict_bow = {'topics' : topic_keywords_bow}
    td_bow = div.score(div_dict_bow)
    
    div_dict_tfidf = {'topics' : topic_keywords_tfidf}
    td_tfidf = div.score(div_dict_tfidf)

    return td_bow,td_tfidf
      
      
def get_irbo(model, token_list,sentences):
    """_summary_: Get irbo from gensim.models.coherencemodel
    : param model: Topic_Model object
    : param token_lists: token list of docs
    : param topics: topics as top words"""
    topic_keywords_bow = get_topic_words(token_list, model.labels_) # get topic words
    topic_keywords_tfidf = get_topic_words_ctfidf(sentences,token_list, model.labels_) # get topic words
    
    rbo = InvertedRBO()
    
    irbo_Dict_bow = {'topics' : topic_keywords_bow}
    irbo_bow = rbo.score(irbo_Dict_bow)
    
    irbo_Dict_tfidf = {'topics' : topic_keywords_tfidf}
    irbo_tfidf = rbo.score(irbo_Dict_tfidf)

    return irbo_bow,irbo_tfidf

def get_rand_index(model, labels_true, token_list):
    
    labels_pred = model.labels_
    labels_pred_len = len(labels_pred)
    labels_true = labels_true[:labels_pred_len]
    labels_true_len = len(labels_true)
    
    scores = rand_score(labels_true, labels_pred)
    return scores

def print_evaluations(model, cluster_model, token_lists, labels, k, method,sentences):
    coherence_cv_bow , coherence_npmi_bow, coherence_cv_tfidf , coherence_npmi_tfidf = get_coherence(model, cluster_model, token_lists,sentences)
    diversity_bow, diversity_tfidf = get_topic_diversity(cluster_model, token_lists, k,sentences)
    irbo_bow, irbo_tfidf = get_irbo(cluster_model, token_lists,sentences)
    # silhoulette = get_silhoulette(model)
    rand_index = get_rand_index(cluster_model, labels, token_lists)
    print("-"*50)
    print("Model       Score")
    print("-"*50)
    print("{} Coherence CV BOW: {}".format(method, coherence_cv_bow))
    print("-"*50)
    print("{} Coherence CV TFIDF: {}".format(method, coherence_cv_tfidf))
    print("-"*50)
    print("{} Coherence NPMI BOW: {}".format(method, coherence_npmi_bow))
    print("-"*50)
    print("{} Coherence NPMI TFIDF: {}".format(method, coherence_npmi_tfidf))
    print("-"*50)
    print("{} Topic Diversity BOW: {}".format(method, diversity_bow))
    print("-"*50)
    print("{} Topic Diversity TFIDF: {}".format(method, diversity_tfidf))
    print("-"*50)
    print("{} IRBO BOW: {}".format(method, irbo_bow))
    print("-"*50)
    print("{} IRBO TFIDF: {}".format(method, irbo_tfidf))
    print("-"*50)
    print("{} Rand Index: {}".format(method, rand_index)) 
    print("-"*50)
    
    # Save the results to a file    
    with open('results_'+model.emebeddingmethod+"_" + model.embed.embedding_model+"_"+model.dimReduction+ "_" +model.clusterMethod + "_"+str(model.k)+'.txt', 'a') as f:
        f.write("-"*50)
        f.write("Model       Score")
        f.write("-"*50)
        f.write("{} Coherence CV: {}".format(method, coherence_cv_bow))
        f.write("-"*50)
        f.write("{} Coherence CV: {}".format(method, coherence_cv_tfidf))
        f.write("-"*50)
        f.write("{} Coherence NPMI: {}".format(method, coherence_npmi_bow))
        f.write("-"*50)
        f.write("{} Coherence NPMI: {}".format(method, coherence_npmi_tfidf))
        f.write("-"*50)
        f.write("{} Topic Diversity: {}".format(method, diversity_bow))
        f.write("-"*50)
        f.write("{} Topic Diversity: {}".format(method, diversity_tfidf))
        f.write("-"*50)
        f.write("{} IRBO: {}".format(method, irbo_bow))
        f.write("-"*50)
        f.write("{} IRBO: {}".format(method, irbo_tfidf))
        f.write("-"*50)
        f.write("{} Rand Index: {}".format(method, rand_index))
        f.write("-"*50)
        
        
    


