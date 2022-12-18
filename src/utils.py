
from preprocess import *
from train import *
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from bertTopic import *
from collections import Counter


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
      
# def get_topic_diversity(model, token_list):
      
    topic_keywords = get_topic_words(token_list, model.cluster_model.labels_) # get topic words

    tm_topics = []  # list of lists of words in each topic
      
      
      # for k,v in topic_keywords.items(): # k is topic number, v is list of tuples (word, prob)
      #     temp = [] # temp list to store words in each topic
      #     for tup in v: # tup is tuple (word, prob)
      #         temp.append(tup[0]) # append word to temp list
      #     tm_topics.append(temp)    # append temp list to tm_topics list

      # unique_words = set()
      # for topic in tm_topics: # topic is list of words
      #     unique_words = unique_words.union(set(topic[:10])) # get top 10 words from each topic
      # td = len(unique_words) / (10 * len(tm_topics)) # calculate topic diversity

      # return td

