from topic2vec import *
from bertTopic import *
from topicModel import *



class Trainers(object):
    def __init__(self,data=None, embeddingmethod=None, dimReduction=None, clusterMethod=None,token_lists=None,gamma=1, k=10, n_epochs=100, batch_size=128, lr=0.001, verbose=1,embeddingmodel=None):
        self.embeddingmethod = embeddingmethod
        self.dimReduction = dimReduction
        self.m_clustering = clusterMethod
        self.embeddingmodel = embeddingmodel
        self.gamma = gamma
        self.k = k # Number of topics
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.vec = {}
        self.AE = None
        self.VAE = None
        self.ldamodel = None
        self.corpus = None
        self.id2word = None
        self.data = data
        self.token_lists = token_lists
                

    def train(self):
        
        if self.embeddingmethod in ["LDA_S_TRANS","LDA_DIFFCSE"]:
            print("Embedding method: ", self.embeddingmethod)
            tm = Topic_Model(k=self.k, embeddingmethod=self.embeddingmethod, dimReduction=self.dimReduction, clusterMethod=self.m_clustering,embeddingmodel=self.embeddingmodel)
            cluster_model=tm.fit(self.data,self.token_lists)
            print("Model used: ", cluster_model)
            return tm, cluster_model    
                
        elif self.embeddingmethod == 'Top2Vec':
            tm = top2vec_train(self.data)
            return tm
        
        elif self.embeddingmethod == 'Bertopic':
            tm = bertopic_train(self.data, self.k)
            
            return tm
            
