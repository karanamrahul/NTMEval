

from lda_bert import *
from topic2vec import *
from lda_umap import *
from HCluster import *



class Trainer(object):
    def __init__(self,data=None, method=None, m_clustering=None,token_lists=None,gamma=1, k=10, n_epochs=100, batch_size=128, lr=0.001, verbose=1):
        self.method = method
        self.m_clustering = m_clustering
        self.gamma = gamma
        self.k = k # Number of topics
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.vec = {}
        self.AE = None
        self.ldamodel = None
        self.corpus = None
        self.id2word = None
        self.data = data
        self.token_lists = token_lists

    def train(self):
        if self.method == 'LDA_BERT':
            tm = Topic_Model(k=self.k, method='LDA_BERT')
            tm.fit(self.data,self.token_lists)
            
            return tm
        
        elif self.method == 'SROBERTA':
            tm = Topic_Model(k=self.k, method='SROBERTA')
            tm.fit(self.data,self.token_lists)
            
            return tm

        elif self.method == 'DIFFCSE':
            tm = Topic_Model(k=self.k, method='DIFFCSE')
            tm.fit(self.data,self.token_lists)
            
            return tm

        elif self.method == 'LDA_DIFFCSE':
            tm = Topic_Model(k=self.k, method='LDA_DIFFCSE')
            tm.fit(self.data,self.token_lists)
            
            return tm
            
        elif self.method == 'Top2Vec':
            tm = top2vec_train(self.data)
            
            return tm
            
