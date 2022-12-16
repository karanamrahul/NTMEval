from top2vec import Top2Vec
from utils import *
np.random.seed(0)
import gensim.corpora as corpora
from gensim.utils import tokenize
from gensim.models import CoherenceModel
TOKENIZERS_PARALLELISM = True





def top2vec_train(sentences, embedding_model='all-MiniLM-L6-v2'):
    """
    Train Top2Vec model
    :param sentences: list of tokenized sentences
    :param embedding_model: word embedding model
    :param documents: list of documents
    :param kwargs: keyword arguments
    :return: Top2Vec model
    """
    model = Top2Vec(sentences, workers=4, embedding_model='all-MiniLM-L6-v2',
    umap_args={'n_neighbors':30, 'n_components':32},
    hdbscan_args={'min_cluster_size':100, 'min_samples':10})
    
    #embedding_batch_size=32
    return model

def evaluate_top2vec(topic_words,sentences):
    tokenized = [list(tokenize(s)) for s in sentences]
    id2word = corpora.Dictionary(tokenized)
    corpus = [id2word.doc2bow(text) for text in tokenized]
    cm = CoherenceModel(topics=[s.tolist() for s in topic_words] ,texts=tokenized, corpus=corpus, dictionary=id2word, coherence='c_v')
       
    print("Model Coherence C_V is:{0}".format(cm.get_coherence()))