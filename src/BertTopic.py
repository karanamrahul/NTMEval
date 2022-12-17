from bertopic import BERTopic
from utils import *
np.random.seed(0)
import gensim.corpora as corpora
from gensim.utils import tokenize
from gensim.models import CoherenceModel
TOKENIZERS_PARALLELISM = True
from umap import UMAP
from sklearn.cluster import AgglomerativeClustering
from hdbscan import HDBSCAN

def bertopic_train(sentences,cluster_count,embedding_model='all-MiniLM-L6-v2', model_type='HDBSCAN'):
  
    umap_model = UMAP(n_neighbors=15, n_components=32, min_dist=0.0, metric='cosine')
    
    if model_type == "HDBSCAN":
        model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    elif model_type == "agglomerative":
        model = AgglomerativeClustering(n_clusters=cluster_count)
        
    model = BERTopic(sentences, embedding_model=embedding_model,
    umap_model=umap_model, hdbscan_model=model)
    
    topics, probs = model.fit_transform(sentences)
    return model

def evaluate_bertopic(topic_words,sentences):
    tokenized = [list(tokenize(s)) for s in sentences]
    id2word = corpora.Dictionary(tokenized)
    corpus = [id2word.doc2bow(text) for text in tokenized]
    cm = CoherenceModel(topics=[s for s in topic_words] ,texts=tokenized, corpus=corpus, dictionary=id2word, coherence='c_v')
       
    print("Model Coherence C_V is:{0}".format(cm.get_coherence()))