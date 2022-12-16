from bertopic import BERTopic
from utils import *
np.random.seed(0)
import gensim.corpora as corpora
from gensim.utils import tokenize
from gensim.models import CoherenceModel
TOKENIZERS_PARALLELISM = True





def bertopic_train(sentences, embedding_model='all-MiniLM-L6-v2'):
    
    docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)
  
    model = BERTopic(sentences, workers=4, embedding_model='all-MiniLM-L6-v2',
    umap_args={'n_neighbors':30, 'n_components':32},
    hdbscan_args={'min_cluster_size':100, 'min_samples':10})
    
    #embedding_batch_size=32
    return model

def evaluate_bertopic(topic_words,sentences):
    tokenized = [list(tokenize(s)) for s in sentences]
    id2word = corpora.Dictionary(tokenized)
    corpus = [id2word.doc2bow(text) for text in tokenized]
    cm = CoherenceModel(topics=[s.tolist() for s in topic_words] ,texts=tokenized, corpus=corpus, dictionary=id2word, coherence='c_v')
       
    print("Model Coherence C_V is:{0}".format(cm.get_coherence()))